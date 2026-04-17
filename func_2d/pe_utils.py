#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
PE-MedSAM2 2D Utils
===================
放置位置: func_2d/pe_utils.py

使用方法:
---------
在训练代码中:

1. 导入:
   from func_2d.pe_utils import apply_pe_to_features, create_pe_modules, create_mal_loss

2. 在 forward 后应用 PE:
   image_embed = feats[-1]
   if args.use_pe:
       image_embed, boundary_map, importance = apply_pe_to_features(image_embed, pe_modules)

3. 使用 MAL 损失:
   if args.use_pe:
       loss, loss_dict = mal_loss(pred, masks, epoch=epoch)
   else:
       loss = lossfunc(pred, masks)

说明:
-----
- PFFE 被消融时，DSA 会通过 fallback 生成的 boundary_map 继续工作，
  确保消融实验真正只移除了 PFFE 这一个模块。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

# 导入 PE 模块
from func_2d.pe_modules import LRA, PFFE, ULA, DSA, MALLoss


def _fallback_boundary_map(features: torch.Tensor) -> torch.Tensor:
    """
    当 PFFE 不存在时，用简单 Sobel 生成 boundary_map 供 DSA 使用。

    这样在 "w/o PFFE" 消融中 DSA 仍能正常工作，
    消融实验才是真正只移除了 PFFE 一个模块。

    Args:
        features: (B, C, H, W) 特征图

    Returns:
        boundary_map: (B, 1, H, W) 边界概率图
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
        boundary_map = torch.sigmoid(5.0 * edge - 2.5)
    return boundary_map


def create_pe_modules(hidden_dim: int, args) -> Dict[str, nn.Module]:
    """
    创建 PE 模块字典。

    Args:
        hidden_dim: 特征维度 (通常是 256)
        args: 配置参数

    Returns:
        modules: 包含所有 PE 模块的字典
    """
    modules = {}

    use_lra  = getattr(args, 'use_lra',  True)
    use_pffe = getattr(args, 'use_pffe', True)
    use_ula  = getattr(args, 'use_ula',  True)
    use_dsa  = getattr(args, 'use_dsa',  True)

    lra_rank        = getattr(args, 'lra_rank', 4)
    ula_compression = getattr(args, 'ula_compression', 16)
    dsa_sparsity    = getattr(args, 'dsa_sparsity', 0.25)
    pffe_scales     = getattr(args, 'pffe_scales', [3, 5, 7])

    if use_lra:
        modules['lra']  = LRA(hidden_dim, rank=lra_rank)
    if use_pffe:
        modules['pffe'] = PFFE(scales=pffe_scales)
    if use_ula:
        modules['ula']  = ULA(hidden_dim, compression_ratio=ula_compression)
    if use_dsa:
        modules['dsa']  = DSA(hidden_dim, num_heads=8, sparsity_ratio=dsa_sparsity)

    # 打印信息
    print("\n" + "=" * 50)
    print("PE Modules Created")
    print("=" * 50)
    total_params = 0
    for name, module in modules.items():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_params += params
        print(f"  {name.upper()}: {params:,} params")
    print(f"  Total: {total_params:,} params")
    print("=" * 50 + "\n")

    return modules


def apply_pe_to_features(
        features: torch.Tensor,
        modules: Dict[str, nn.Module]
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    将 PE 模块应用到特征上。

    Args:
        features: 输入特征 (B, C, H, W)
        modules: PE 模块字典

    Returns:
        enhanced_features: 增强后的特征
        boundary_map: 边界图 (可选)
        importance: 重要性图 (可选)
    """
    boundary_map = None
    importance = None

    # LRA: 低秩域适配
    if 'lra' in modules:
        features = modules['lra'](features)

    # PFFE: 无参数特征增强
    if 'pffe' in modules:
        features, boundary_map = modules['pffe'](features, return_boundary_map=True)
    else:
        # 即使没有 PFFE，也生成 boundary_map 供 DSA 使用
        # 使用 no_grad 的简单 Sobel，不影响 PFFE 消融的公平性
        if 'dsa' in modules:
            boundary_map = _fallback_boundary_map(features)

    # ULA: 超轻量适配器
    if 'ula' in modules:
        features = modules['ula'](features)

    # DSA: 动态稀疏注意力
    if 'dsa' in modules:
        features, importance = modules['dsa'](features, boundary_map)

    return features, boundary_map, importance


def create_mal_loss(args, device) -> MALLoss:
    """
    创建 MAL 损失函数。

    注意: MALLoss 默认参数已与 cfg_pe.py 统一
          (lambda_bce=0.5, lambda_boundary=0.3)
    """
    return MALLoss(
        lambda_dice=getattr(args, 'lambda_dice', 1.0),
        lambda_bce=getattr(args, 'lambda_bce', 0.5),
        lambda_boundary=getattr(args, 'lambda_boundary', 0.3),
        warmup_epochs=getattr(args, 'mal_warmup', 10)
    ).to(device)


def get_pe_parameters(modules: Dict[str, nn.Module]) -> list:
    """
    获取所有 PE 模块的可训练参数。
    """
    params = []
    for module in modules.values():
        params.extend([p for p in module.parameters() if p.requires_grad])
    return params