#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
PE-MedSAM2 2D Modules (论文对齐版)
====================================
放置位置: func_2d/pe_modules.py

包含所有 2D PE 模块:
- LRA:  Low-Rank Adapter (低秩域适配器)
- PFFE: Parameter-Free Feature Enhancement (无参数特征增强)
- ULA:  Ultra-Lightweight Adapter (超轻量适配器)
- DSA:  Dynamic Sparse Attention (动态稀疏注意力)
- MALLoss: Multi-objective Adaptive Loss (多目标自适应损失)

设计原则:
  [LRA]  低秩残差投影 + GELU 激活 + 可学习 α
  [PFFE] 固定 Sobel/Laplacian 多尺度 + Z-score 归一化 (零可学习参数)
  [ULA]  深度可分离 DWConv + Squeeze-Excitation + 可学习 α
  [DSA]  PFFE 边界先验 + top-K 稀疏注意力 + 仅对 top-K token 做 norm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


# =============================================================================
# LRA: Low-Rank Adapter
# =============================================================================

class LRA(nn.Module):
    """
    Low-Rank Adapter using LoRA-style low-rank decomposition.

    论文公式:
        F_down  = F^token · W_down
        F_adapt = σ(F_down) · W_up      ← σ = GELU
        F_out   = F^token + α · F_adapt
    """

    def __init__(self, in_features: int, rank: int = 4, alpha_init: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.rank = rank

        self.W_down = nn.Linear(in_features, rank, bias=False)
        self.activation = nn.GELU()
        self.W_up = nn.Linear(rank, in_features, bias=False)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        nn.init.kaiming_uniform_(self.W_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.W_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_spatial = len(x.shape) == 4

        if is_spatial:
            B, C, H, W = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
            delta = self.W_up(self.activation(self.W_down(x_flat)))
            delta = delta.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            delta = self.W_up(self.activation(self.W_down(x)))

        return x + self.alpha * delta


# =============================================================================
# PFFE: Parameter-Free Feature Enhancement
# =============================================================================

class PFFE(nn.Module):
    """
    Parameter-Free Feature Enhancement using multi-scale Sobel and Laplacian.

    论文公式:
        B = (1/|S|) Σ E^(s) + |F_avg * L|      (S={3,5,7})
        A_boundary = σ(γ · (B - μ_B) / σ_B + β)
        F_enhanced = F ⊙ (1 + A_boundary)
    """

    def __init__(self, scales: list = None, gamma: float = 5.0, beta: float = -2.5):
        super().__init__()
        self.scales = scales or [3, 5, 7]
        self.gamma = gamma
        self.beta = beta

        for scale in self.scales:
            sobel_x, sobel_y = self._create_sobel_kernels(scale)
            self.register_buffer(f'sobel_x_{scale}', sobel_x)
            self.register_buffer(f'sobel_y_{scale}', sobel_y)

        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer('laplacian', laplacian.unsqueeze(0).unsqueeze(0))

    def _create_sobel_kernels(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if size == 3:
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        elif size == 5:
            sobel_x = torch.tensor([
                [-1, -2, 0, 2, 1], [-4, -8, 0, 8, 4], [-6, -12, 0, 12, 6],
                [-4, -8, 0, 8, 4], [-1, -2, 0, 2, 1]
            ], dtype=torch.float32)
        elif size == 7:
            sobel_x = torch.tensor([
                [-1, -4, -5, 0, 5, 4, 1], [-6, -24, -30, 0, 30, 24, 6],
                [-15, -60, -75, 0, 75, 60, 15], [-20, -80, -100, 0, 100, 80, 20],
                [-15, -60, -75, 0, 75, 60, 15], [-6, -24, -30, 0, 30, 24, 6],
                [-1, -4, -5, 0, 5, 4, 1]
            ], dtype=torch.float32)
        else:
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)

        sobel_y = sobel_x.t()
        return sobel_x.unsqueeze(0).unsqueeze(0), sobel_y.unsqueeze(0).unsqueeze(0)

    def _compute_boundary_map(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 4:
            x_gray = x.mean(dim=1, keepdim=True)
        else:
            x_gray = x

        edge_maps = []
        for scale in self.scales:
            sobel_x = getattr(self, f'sobel_x_{scale}')
            sobel_y = getattr(self, f'sobel_y_{scale}')
            padding = scale // 2

            grad_x = F.conv2d(x_gray, sobel_x, padding=padding)
            grad_y = F.conv2d(x_gray, sobel_y, padding=padding)
            edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
            edge_maps.append(edge)

        laplacian_edge = torch.abs(F.conv2d(x_gray, self.laplacian, padding=1))
        edge_maps.append(laplacian_edge)

        combined = torch.cat(edge_maps, dim=1).mean(dim=1, keepdim=True)

        # A = σ(γ · (B - μ_B) / σ_B + β)
        mu = combined.mean(dim=(2, 3), keepdim=True)
        sigma = combined.std(dim=(2, 3), keepdim=True) + 1e-6
        normalized = (combined - mu) / sigma
        boundary_map = torch.sigmoid(self.gamma * normalized + self.beta)

        return boundary_map

    def forward(self, x: torch.Tensor, return_boundary_map: bool = False):
        boundary_map = self._compute_boundary_map(x)
        attention = 1.0 + boundary_map
        out = x * attention

        if return_boundary_map:
            return out, boundary_map
        return out


# =============================================================================
# ULA: Ultra-Lightweight Adapter
# =============================================================================

class ULA(nn.Module):
    """
    Ultra-Lightweight Adapter: depthwise conv + SE.

    Spatial pathway: GELU(DWConv_{3×3}(F))  →  9C 参数
    Channel pathway: σ(W_up · GELU(W_down · GAP(F)))  →  2C²/r 参数
    Fusion: F + α · (F_spatial ⊙ F_channel)

    与标准 adapter 相比减少 67% 参数 (C=256, r=16: 10,496 vs 32,768)
    """

    def __init__(self, dim: int, compression_ratio: int = 16, kernel_size: int = 3,
                 alpha_init: float = 0.1):
        super().__init__()
        reduced_dim = max(dim // compression_ratio, 8)

        self.dw_conv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.spatial_act = nn.GELU()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(dim, reduced_dim), nn.GELU(),
            nn.Linear(reduced_dim, dim), nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.norm = nn.LayerNorm(dim)

        nn.init.normal_(self.dw_conv.weight, std=0.01)
        nn.init.zeros_(self.dw_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        spatial = self.spatial_act(self.dw_conv(x))
        channel = self.se(x).view(B, C, 1, 1)
        out = x + self.alpha * (spatial * channel)
        out = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out


# =============================================================================
# DSA: Dynamic Sparse Attention
# =============================================================================

class DSA(nn.Module):
    """
    Dynamic Sparse Attention.

    Importance scoring: S_final = S + λ·B  (加性融合 PFFE 边界先验)
    Top-K selection: K = ⌊ρ·N⌋ (ρ=0.25, 减少 93.75% 注意力计算)
    Sparse attention: 仅在 K 个 token 之间计算注意力, 再 scatter 回原位置
    """

    def __init__(self, dim: int, num_heads: int = 8, sparsity_ratio: float = 0.25,
                 boundary_weight: float = 0.5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sparsity_ratio = sparsity_ratio
        self.boundary_weight = boundary_weight

        self.importance_net = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1), nn.GELU(),
            nn.Conv2d(dim // 4, 1, 1), nn.Sigmoid()
        )
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, boundary_map: torch.Tensor = None):
        B, C, H, W = x.shape
        N, K = H * W, max(int(H * W * self.sparsity_ratio), 1)

        importance = self.importance_net(x)
        if boundary_map is not None:
            # S_final = S + λ·B (加性融合)
            importance = importance + self.boundary_weight * boundary_map

        _, top_idx = torch.topk(importance.view(B, N), K, dim=1)
        x_flat = x.permute(0, 2, 3, 1).reshape(B, N, C)
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, K)
        x_sparse = x_flat[batch_idx, top_idx]

        q = self.q_proj(x_sparse).reshape(B, K, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x_sparse).reshape(B, K, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x_sparse).reshape(B, K, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out_sparse = self.out_proj((attn @ v).permute(0, 2, 1, 3).reshape(B, K, C))

        # 仅对 top-K token 做残差 + norm
        enhanced_sparse = self.norm(out_sparse + x_sparse)
        out_full = x_flat.clone()
        out_full[batch_idx, top_idx] = enhanced_sparse

        out = out_full.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return out, importance


# =============================================================================
# MAL: Multi-objective Adaptive Loss
# =============================================================================

class MALLoss(nn.Module):
    """
    Multi-objective Adaptive Loss (MAL)

        L = λ₁·L_dice + λ₂·L_bce + λ₃·L_boundary

    其中 L_boundary 对形态学提取的边界像素集合进行 BCE 监督,
    λ₃ 通过 warmup 策略渐进上升, 让模型先建立稳定的区域预测.
    """

    def __init__(self, lambda_dice=1.0, lambda_bce=0.5, lambda_boundary=0.3,
                 warmup_epochs=10):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce
        self.lambda_boundary = lambda_boundary
        self.warmup_epochs = warmup_epochs

    def _dice_loss(self, pred, target):
        pred_sig = torch.sigmoid(pred)
        smooth = 1e-5
        intersection = (pred_sig * target).sum(dim=(2, 3))
        union = pred_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    def _bce_loss(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target)

    def _boundary_loss(self, pred, target, kernel_size=3):
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=target.device)
        padding = kernel_size // 2

        dilated = (F.conv2d(target, kernel, padding=padding) > 0).float()
        eroded = (F.conv2d(target, kernel, padding=padding) >= kernel_size ** 2).float()
        boundary = (dilated - eroded).clamp(0, 1)

        boundary_weight = boundary + 1e-6

        return F.binary_cross_entropy_with_logits(
            pred * boundary_weight,
            target * boundary_weight,
            reduction='mean'
        )

    def forward(self, pred, target, epoch=0):
        warmup_factor = min(1.0, epoch / max(1, self.warmup_epochs))

        dice_loss = self._dice_loss(pred, target)
        bce_loss = self._bce_loss(pred, target)
        boundary_loss = self._boundary_loss(pred, target)

        total_loss = (
                self.lambda_dice * dice_loss +
                self.lambda_bce * bce_loss +
                self.lambda_boundary * warmup_factor * boundary_loss
        )

        loss_dict = {
            'dice': dice_loss.item(),
            'bce': bce_loss.item(),
            'boundary': boundary_loss.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict


class ImportanceLoss(nn.Module):
    """Auxiliary loss for DSA importance learning."""

    def __init__(self):
        super().__init__()

    def forward(self, importance, mask):
        kernel = torch.ones(1, 1, 3, 3, device=mask.device)
        dilated = (F.conv2d(mask, kernel, padding=1) > 0).float()
        eroded = (F.conv2d(mask, kernel, padding=1) >= 9).float()
        boundary = (dilated - eroded).clamp(0, 1)
        return F.binary_cross_entropy_with_logits(importance, boundary)


# =============================================================================
# Boundary Map Fallback (用于消融实验, PFFE 被移除时仍需 boundary_map 供 DSA 使用)
# =============================================================================

def compute_boundary_map_fallback(features: torch.Tensor) -> torch.Tensor:
    """
    当 PFFE 被消融时，为 DSA 生成 boundary_map。

    用简单 Sobel 3×3 + 归一化，不增强特征，不引入参数。
    no_grad 确保不参与反向传播，不影响消融公平性。
    """
    with torch.no_grad():
        x_gray = features.mean(dim=1, keepdim=True)
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=features.dtype, device=features.device
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)

        gx = F.conv2d(x_gray, sobel_x, padding=1)
        gy = F.conv2d(x_gray, sobel_y, padding=1)
        edge = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

        mu = edge.mean(dim=(2, 3), keepdim=True)
        sigma = edge.std(dim=(2, 3), keepdim=True) + 1e-6
        boundary_map = torch.sigmoid(5.0 * (edge - mu) / sigma - 2.5)

    return boundary_map


# =============================================================================
# PE-MedSAM2 Wrapper
# =============================================================================

class PEMedSAM2(nn.Module):
    """PE-MedSAM2: Integrating PE modules with MedSAM2."""

    def __init__(self, base_model, args, freeze_encoder=True,
                 use_lra=True, use_pffe=True, use_ula=True, use_dsa=True,
                 lra_rank=4, ula_compression=16, dsa_sparsity=0.25, pffe_scales=None):
        super().__init__()

        self.base_model = base_model
        self.args = args
        self.use_lra = use_lra
        self.use_pffe = use_pffe
        self.use_ula = use_ula
        self.use_dsa = use_dsa
        self.hidden_dim = base_model.hidden_dim

        if freeze_encoder:
            for n, p in base_model.named_parameters():
                if 'image_encoder' in n:
                    p.requires_grad = False
            print("[PE-MedSAM2] Image encoder frozen")

        if pffe_scales is None:
            pffe_scales = [3, 5, 7]

        if use_lra:  self.lra  = LRA(self.hidden_dim, rank=lra_rank)
        if use_pffe: self.pffe = PFFE(scales=pffe_scales)
        if use_ula:  self.ula  = ULA(self.hidden_dim, compression_ratio=ula_compression)
        if use_dsa:  self.dsa  = DSA(self.hidden_dim, num_heads=8, sparsity_ratio=dsa_sparsity)

        self._print_info()

    def _print_info(self):
        print("\n" + "=" * 50)
        print("PE-MedSAM2 Configuration")
        print("=" * 50)
        total = 0
        for name in ['lra', 'pffe', 'ula', 'dsa']:
            if hasattr(self, name):
                p = sum(x.numel() for x in getattr(self, name).parameters() if x.requires_grad)
                total += p
                print(f"  {name.upper()}: {p:,} params")
        print(f"  Total PE: {total:,}")
        print("=" * 50 + "\n")

    def apply_pe_modules(self, features):
        """
        PFFE 消融时仍为 DSA 提供 boundary_map (fallback)。
        """
        boundary_map = None
        importance = None

        # LRA
        if self.use_lra and hasattr(self, 'lra'):
            features = self.lra(features)

        # PFFE
        if self.use_pffe and hasattr(self, 'pffe'):
            features, boundary_map = self.pffe(features, return_boundary_map=True)
        else:
            # fallback: 不增强特征, 仅生成 boundary_map 供 DSA 使用
            boundary_map = compute_boundary_map_fallback(features)

        # ULA
        if self.use_ula and hasattr(self, 'ula'):
            features = self.ula(features)

        # DSA
        if self.use_dsa and hasattr(self, 'dsa'):
            features, importance = self.dsa(features, boundary_map)

        return features, boundary_map, importance

    def forward_image(self, images):
        return self.base_model.forward_image(images)

    def _prepare_backbone_features(self, x):
        return self.base_model._prepare_backbone_features(x)

    def _encode_new_memory(self, *a, **k):
        return self.base_model._encode_new_memory(*a, **k)

    def memory_attention(self, *a, **k):
        return self.base_model.memory_attention(*a, **k)

    @property
    def sam_prompt_encoder(self):
        return self.base_model.sam_prompt_encoder

    @property
    def sam_mask_decoder(self):
        return self.base_model.sam_mask_decoder


if __name__ == '__main__':
    print("Testing PE 2D modules...")
    x = torch.randn(2, 256, 16, 16)

    for name, mod in [('LRA', LRA(256)), ('PFFE', PFFE()),
                      ('ULA', ULA(256)), ('DSA', DSA(256))]:
        if name == 'PFFE':
            out, bmap = mod(x, return_boundary_map=True)
            print(f"  boundary_map: mean={bmap.mean():.4f}, max={bmap.max():.4f}, "
                  f"min={bmap.min():.4f}, std={bmap.std():.4f}")
        elif name == 'DSA':
            out, _ = mod(x, bmap)
        else:
            out = mod(x)
        print(f"{name}: {x.shape} -> {out.shape}")

    # 测试 fallback
    bmap_fb = compute_boundary_map_fallback(x)
    print(f"\nFallback boundary_map: shape={bmap_fb.shape}, "
          f"mean={bmap_fb.mean():.4f}, max={bmap_fb.max():.4f}")

    # 验证参数量
    print("\n--- Parameter Count ---")
    for name, mod in [('LRA', LRA(256)), ('PFFE', PFFE()),
                      ('ULA', ULA(256)), ('DSA', DSA(256))]:
        p = sum(x.numel() for x in mod.parameters() if x.requires_grad)
        print(f"  {name}: {p:,} params")

    print("\nAll PE module tests passed!")