#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
measure_efficiency.py - PE-MedSAM2 模型效率测量 (表8 & 表9)
===========================================================
测量内容:
  1. 参数量 (total / trainable / 新增 PE 模块细分)
  2. FLOPs (encoder 用 fvcore, PE 模块手动计算)
  3. 推理时间 (warmup + 多次测量取均值)
  4. 分段计时 (encoder / PE / decoder)

PE 模块: LRA + PFFE + ULA + DSA

用法:
  cd /root/autodl-tmp/PE-MedSAM2
  python measure_efficiency.py
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PE_PATH = '/root/autodl-tmp/PE-MedSAM2'
sys.path.insert(0, PE_PATH)

IMAGE_SIZE = 1024


# ============================================================
# 1. 参数量统计
# ============================================================

def count_params_by_component(model):
    components = {
        'image_encoder': {'total': 0, 'trainable': 0},
        'memory_attention': {'total': 0, 'trainable': 0},
        'memory_encoder': {'total': 0, 'trainable': 0},
        'prompt_encoder': {'total': 0, 'trainable': 0},
        'mask_decoder': {'total': 0, 'trainable': 0},
        'other': {'total': 0, 'trainable': 0},
    }
    for name, param in model.named_parameters():
        num = param.numel()
        matched = False
        for comp in ['image_encoder', 'memory_attention', 'memory_encoder',
                     'sam_prompt_encoder', 'sam_mask_decoder']:
            if comp in name:
                key = comp.replace('sam_', '')
                if key not in components:
                    key = 'other'
                components[key]['total'] += num
                if param.requires_grad:
                    components[key]['trainable'] += num
                matched = True
                break
        if not matched:
            components['other']['total'] += num
            if param.requires_grad:
                components['other']['trainable'] += num
    return components


def count_pe_detail(pe_modules):
    details = {}
    total = 0
    for name, module in pe_modules.items():
        if isinstance(module, nn.Module):
            p = sum(x.numel() for x in module.parameters())
            details[name.upper()] = p
            total += p
        else:
            details[name.upper()] = 0
    details['_total'] = total
    return details


# ============================================================
# 2. FLOPs 计算
# ============================================================

def measure_flops_fvcore(model, input_shape=(1, 3, 1024, 1024), device='cuda'):
    """方案 A: 用 fvcore (SAM2 项目自带依赖, 支持最好)"""
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table
        dummy = torch.randn(*input_shape, device=device)
        flops_analyzer = FlopCountAnalysis(model, dummy)
        flops_analyzer.unsupported_ops_warnings(False)
        flops_analyzer.uncalled_modules_warnings(False)
        total_flops = flops_analyzer.total()
        return total_flops, 'fvcore'
    except ImportError:
        return None, 'fvcore not installed'
    except Exception as e:
        return None, f'fvcore error: {e}'


def measure_flops_manual_encoder(model_name='hiera_small'):
    """方案 B: 手动计算 Hiera encoder FLOPs (理论值)

    SAM2 用的是 Hiera-S (sam2.1_hiera_small):
      - Stages: [1, 2, 11, 2] blocks
      - Dims: [96, 192, 384, 768]
      - Heads: [1, 2, 4, 8]
      - Window sizes 变化

    每个 block 的 FLOPs ≈
      Attention: 4*N*D² (QKV+proj) + 2*N²*D (attention matmul)
      MLP: 8*N*D² (两层 MLP, expansion=4)
    其中 N=token 数, D=dim
    """
    stages = [1, 2, 11, 2]
    dims = [96, 192, 384, 768]

    # 输入 1024x1024, patch_size=16 in Hiera → 初始 64x64 = 4096 tokens
    # Stage 间有 2x2 pooling → tokens: 4096, 1024, 256, 64
    tokens_per_stage = [4096, 1024, 256, 64]

    total_flops = 0

    for stage_idx, (n_blocks, dim, n_tokens) in enumerate(
            zip(stages, dims, tokens_per_stage)):
        for _ in range(n_blocks):
            # Attention: QKV projection + output projection
            attn_proj = 4 * n_tokens * dim * dim * 2  # ×2 for multiply-add
            # 对于 windowed attention, 实际 ≈ 2 * n_tokens * window_size² * dim
            window_area = min(n_tokens, 8 * 8)
            attn_matmul_windowed = 2 * n_tokens * window_area * dim * 2

            # MLP: dim → 4*dim → dim
            mlp_flops = 2 * n_tokens * dim * (4 * dim) * 2

            block_flops = attn_proj + attn_matmul_windowed + mlp_flops
            total_flops += block_flops

        # Pooling between stages (relatively small)
        if stage_idx < len(stages) - 1:
            total_flops += n_tokens * dim * dims[stage_idx + 1] * 2

    # Patch embedding: 1024x1024x3 → 64x64x96
    patch_embed_flops = (1024 * 1024 * 3 * 96 * 16 * 16) // (16 * 16)
    total_flops += patch_embed_flops * 2

    return total_flops


def measure_flops_lra(dim=256, rank=4, n_tokens=4096):
    """LRA FLOPs: 两个低秩矩阵 + GELU 激活"""
    # W_down: N × D × r × 2
    # W_up:   N × r × D × 2
    return n_tokens * (dim * rank + rank * dim) * 2


def measure_flops_pffe(n_tokens=4096, channels=256):
    """PFFE FLOPs: Sobel + Laplacian 卷积 (非常小)"""
    # 3x3 Sobel: H×W × 9 × 2
    # 5x5 Sobel: H×W × 25 × 2
    # 7x7 Sobel: H×W × 49 × 2
    # Laplacian: H×W × 9 × 2
    h = w = int(n_tokens ** 0.5)  # 64
    return h * w * (9 + 9 + 25 + 25 + 49 + 49 + 9) * 2


def measure_flops_ula(dim=256, compression_ratio=16, kernel_size=3, n_tokens=4096):
    """ULA FLOPs: DWConv + SE (Squeeze-and-Excitation)

    DWConv: depthwise 3×3 → N × k² × C (per-channel)
    SE:
      AdaptiveAvgPool: ~N×C (reduction)
      Linear(C, C//r): C × C//r × 2
      Linear(C//r, C): C//r × C × 2
    Element-wise: N × C × 2 (spatial * channel)
    LayerNorm: N × C × 4 (approx)
    """
    reduced_dim = max(dim // compression_ratio, 8)  # =16

    # DWConv: depthwise, each channel does k×k conv over H×W
    dwconv_flops = n_tokens * kernel_size * kernel_size * dim * 2

    # SE: pool 是 ~N*C adds, 然后两个 FC 层 (operate on 1×C, not N×C)
    se_flops = (dim * reduced_dim + reduced_dim * dim) * 2

    # Element-wise multiply (spatial * channel) + residual add
    elemwise_flops = n_tokens * dim * 2

    # LayerNorm
    norm_flops = n_tokens * dim * 4

    return dwconv_flops + se_flops + elemwise_flops + norm_flops


def measure_flops_dsa(dim=256, num_heads=8, sparsity_ratio=0.25, n_tokens=4096):
    """DSA FLOPs: Dynamic Sparse Attention

    importance_net:
      Conv2d(dim, dim//4, 1): N × D × D/4 × 2
      Conv2d(dim//4, 1, 1):   N × D/4 × 1 × 2
    QKV projections on K tokens: 3 × K × D × D × 2
    Attention matmul: K × K × D × 2 (QK^T) + K × K × D × 2 (attn×V)
    out_proj: K × D × D × 2
    LayerNorm on K tokens: K × D × 4
    """
    K = max(int(n_tokens * sparsity_ratio), 1)  # 1024

    # importance_net: two 1x1 convs
    imp_flops = n_tokens * dim * (dim // 4) * 2 + n_tokens * (dim // 4) * 1 * 2

    # QKV on sparse tokens
    qkv_flops = 3 * K * dim * dim * 2

    # Attention: QK^T + softmax×V
    attn_flops = 2 * K * K * dim * 2

    # Output projection
    out_flops = K * dim * dim * 2

    # LayerNorm on K tokens
    norm_flops = K * dim * 4

    return imp_flops + qkv_flops + attn_flops + out_flops + norm_flops


# ============================================================
# 3. 推理时间测量
# ============================================================

def measure_inference_time(model, pe_modules=None, num_warmup=10, num_runs=50, device='cuda'):
    model.eval()
    if pe_modules:
        for m in pe_modules.values():
            if isinstance(m, nn.Module):
                m.eval()

    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device, dtype=torch.float32)
    pt = torch.tensor([[[512.0, 512.0]]], device=device)
    pt_label = torch.ones(1, 1, device=device, dtype=torch.int)
    feat_sizes = [(IMAGE_SIZE // 4,) * 2, (IMAGE_SIZE // 8,) * 2, (IMAGE_SIZE // 16,) * 2]

    def run_forward():
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            backbone_out = model.forward_image(dummy)
            _, vision_feats, vision_pos_embeds, _ = model._prepare_backbone_features(backbone_out)
            B = vision_feats[-1].size(1)
            vision_feats[-1] = vision_feats[-1] + torch.zeros(1, B, model.hidden_dim, device=device)
            vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.zeros(1, B, model.hidden_dim, device=device)
            feats = [feat.permute(1, 2, 0).view(B, -1, *fs)
                     for feat, fs in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
            image_embed = feats[-1]
            if pe_modules is not None:
                from func_2d.pe_utils import apply_pe_to_features
                image_embed, _, _ = apply_pe_to_features(image_embed, pe_modules)
            high_res_feats = feats[:-1]
            se, de = model.sam_prompt_encoder(points=(pt, pt_label), boxes=None, masks=None, batch_size=B)
            low_res_masks, _, _, _ = model.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se, dense_prompt_embeddings=de,
                multimask_output=False, repeat_image=False,
                high_res_features=high_res_feats)
            F.interpolate(low_res_masks, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)

    for _ in range(num_warmup):
        run_forward()
    torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        run_forward()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

    t = np.array(times)
    return {'mean': float(np.mean(t)), 'std': float(np.std(t)), 'median': float(np.median(t))}


# ============================================================
# 4. 分段计时
# ============================================================

def measure_segmented_time(model, pe_modules=None, num_runs=30, device='cuda'):
    """分段测时间: encoder / PE / decoder 各花多少"""
    model.eval()
    if pe_modules:
        for m in pe_modules.values():
            if isinstance(m, nn.Module): m.eval()

    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device, dtype=torch.float32)
    pt = torch.tensor([[[512.0, 512.0]]], device=device)
    pt_label = torch.ones(1, 1, device=device, dtype=torch.int)
    feat_sizes = [(IMAGE_SIZE // 4,) * 2, (IMAGE_SIZE // 8,) * 2, (IMAGE_SIZE // 16,) * 2]

    enc_times, pe_times, dec_times = [], [], []

    # warmup
    for _ in range(5):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            bo = model.forward_image(dummy)
    torch.cuda.synchronize()

    for _ in range(num_runs):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Encoder
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            backbone_out = model.forward_image(dummy)
            e.record()
            torch.cuda.synchronize()
            enc_times.append(s.elapsed_time(e))

            _, vision_feats, vision_pos_embeds, _ = model._prepare_backbone_features(backbone_out)
            B = vision_feats[-1].size(1)
            vision_feats[-1] += torch.zeros(1, B, model.hidden_dim, device=device)
            vision_pos_embeds[-1] += torch.zeros(1, B, model.hidden_dim, device=device)
            feats = [feat.permute(1, 2, 0).view(B, -1, *fs)
                     for feat, fs in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
            image_embed = feats[-1]

            # PE
            if pe_modules is not None:
                from func_2d.pe_utils import apply_pe_to_features
                s2 = torch.cuda.Event(enable_timing=True)
                e2 = torch.cuda.Event(enable_timing=True)
                s2.record()
                image_embed, _, _ = apply_pe_to_features(image_embed, pe_modules)
                e2.record()
                torch.cuda.synchronize()
                pe_times.append(s2.elapsed_time(e2))

            high_res_feats = feats[:-1]

            # Decoder
            s3 = torch.cuda.Event(enable_timing=True)
            e3 = torch.cuda.Event(enable_timing=True)
            s3.record()
            se, de = model.sam_prompt_encoder(points=(pt, pt_label), boxes=None, masks=None, batch_size=B)
            low_res_masks, _, _, _ = model.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se, dense_prompt_embeddings=de,
                multimask_output=False, repeat_image=False,
                high_res_features=high_res_feats)
            F.interpolate(low_res_masks, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
            e3.record()
            torch.cuda.synchronize()
            dec_times.append(s3.elapsed_time(e3))

    return {
        'encoder': np.mean(enc_times),
        'pe': np.mean(pe_times) if pe_times else 0,
        'decoder': np.mean(dec_times),
    }


# ============================================================
# Main
# ============================================================

def main():
    import cfg_pe as cfg
    args = cfg.parse_args()
    for k, v in {'lra_rank': 4, 'pffe_scales': [3, 5, 7],
                 'use_ula': True, 'use_dsa': True}.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    device = torch.device('cuda', args.gpu_device)

    # GPU 信息
    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1e9

    print("=" * 75)
    print("  PE-MedSAM2 模型效率测量 (表8 & 表9)")
    print(f"  GPU: {gpu_name} ({gpu_mem:.0f}GB)")
    print(f"  输入: {IMAGE_SIZE}×{IMAGE_SIZE}")
    print("=" * 75)

    # ============================================
    # A. MedSAM2 Baseline
    # ============================================
    print(f"\n{'─' * 75}")
    print("  [A] MedSAM2 (Baseline)")
    print(f"{'─' * 75}")

    from func_2d.utils import get_network
    net_base = get_network(args, args.net, use_gpu=args.gpu, gpu_device=device, distribution=args.distributed)
    for name, param in net_base.named_parameters():
        if 'image_encoder' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    comp_base = count_params_by_component(net_base)
    total_base = sum(c['total'] for c in comp_base.values())
    trainable_base = sum(c['trainable'] for c in comp_base.values())

    print(f"\n  参数分布:")
    print(f"  {'组件':<25} {'总参数':>12} {'可训练':>12} {'冻结':>12}")
    print(f"  {'─' * 63}")
    for cn, cc in comp_base.items():
        if cc['total'] > 0:
            print(f"  {cn:<25} {cc['total']:>12,} {cc['trainable']:>12,} {cc['total'] - cc['trainable']:>12,}")
    print(f"  {'─' * 63}")
    print(f"  {'合计':<25} {total_base:>12,} {trainable_base:>12,} {total_base - trainable_base:>12,}")
    print(f"\n  总参数: {total_base / 1e6:.2f}M, 可训练: {trainable_base / 1e6:.2f}M")

    # FLOPs - 用 fvcore
    print(f"\n  FLOPs (Image Encoder)...")
    flops_enc, method = measure_flops_fvcore(net_base.image_encoder, (1, 3, IMAGE_SIZE, IMAGE_SIZE), device)
    if flops_enc is not None and flops_enc > 1000:  # 合理性检查
        print(f"  Image Encoder: {flops_enc / 1e9:.2f} GFLOPs ({method})")
        enc_flops_val = flops_enc
    else:
        print(f"  fvcore 结果异常 ({flops_enc}, {method}), 使用手动估算...")
        enc_flops_val = measure_flops_manual_encoder('hiera_small')
        print(f"  Image Encoder: ~{enc_flops_val / 1e9:.1f} GFLOPs (手动估算)")

    # 推理时间
    print(f"\n  推理时间 (warmup=10, runs=50)...")
    timing_base = measure_inference_time(net_base, device=device)
    print(f"  推理时间: {timing_base['mean']:.1f} ± {timing_base['std']:.1f} ms")

    # 分段计时
    seg_base = measure_segmented_time(net_base, device=device)
    print(f"  分段: encoder={seg_base['encoder']:.1f}ms, decoder={seg_base['decoder']:.1f}ms")

    del net_base
    torch.cuda.empty_cache()

    # ============================================
    # B. PE-MedSAM2 (Ours)
    # ============================================
    print(f"\n{'─' * 75}")
    print("  [B] PE-MedSAM2 (Ours)")
    print(f"{'─' * 75}")

    from func_2d.pe_utils import create_pe_modules
    net_pe = get_network(args, args.net, use_gpu=args.gpu, gpu_device=device, distribution=args.distributed)
    for name, param in net_pe.named_parameters():
        if 'image_encoder' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    args.use_pe = True
    args.use_lra = True
    args.use_pffe = True
    args.use_ula = True
    args.use_dsa = True
    args.lra_rank = 4
    args.pffe_scales = [3, 5, 7]

    pe_modules = create_pe_modules(net_pe.hidden_dim, args)
    for name, module in pe_modules.items():
        if isinstance(module, nn.Module):
            pe_modules[name] = module.to(device)

    total_sam2 = sum(c['total'] for c in count_params_by_component(net_pe).values())
    pe_detail = count_pe_detail(pe_modules)
    pe_total = pe_detail['_total']
    total_pe = total_sam2 + pe_total

    print(f"\n  PE-MedSAM2 新增模块:")
    print(f"  {'模块':<15} {'参数量':>12} {'占比':>8}")
    print(f"  {'─' * 37}")
    for n, p in pe_detail.items():
        if n == '_total': continue
        pct = (p / pe_total * 100) if pe_total > 0 else 0
        print(f"  {n:<15} {p:>12,} {pct:>7.2f}%")
    print(f"  {'─' * 37}")
    print(f"  {'总新增':<15} {pe_total:>12,} {'100.00%':>8}")
    print(f"\n  总参数: {total_pe / 1e6:.2f}M (SAM2 {total_sam2 / 1e6:.2f}M + PE {pe_total / 1e3:.1f}K)")

    # FLOPs - PE 模块手动计算
    dim = net_pe.hidden_dim  # 256
    n_tokens = (IMAGE_SIZE // 16) ** 2  # 4096

    flops_lra = measure_flops_lra(dim, rank=4, n_tokens=n_tokens)
    flops_pffe = measure_flops_pffe(n_tokens, dim)
    flops_ula = measure_flops_ula(dim, compression_ratio=16, kernel_size=3, n_tokens=n_tokens)
    flops_dsa = measure_flops_dsa(dim, num_heads=8, sparsity_ratio=0.25, n_tokens=n_tokens)
    flops_pe_total = flops_lra + flops_pffe + flops_ula + flops_dsa

    print(f"\n  PE 模块 FLOPs:")
    print(f"    LRA:   {flops_lra / 1e6:.1f} MFLOPs")
    print(f"    PFFE:  {flops_pffe / 1e6:.2f} MFLOPs (parameter-free)")
    print(f"    ULA:   {flops_ula / 1e6:.1f} MFLOPs (DWConv + SE)")
    print(f"    DSA:   {flops_dsa / 1e9:.2f} GFLOPs (sparse attention, top-{max(int(n_tokens * 0.25), 1)} tokens)")
    print(f"    总计:  {flops_pe_total / 1e9:.2f} GFLOPs")

    total_flops = enc_flops_val + flops_pe_total
    print(
        f"\n  总FLOPs: ~{total_flops / 1e9:.1f}G (encoder {enc_flops_val / 1e9:.1f}G + PE {flops_pe_total / 1e9:.2f}G)")

    # 推理时间
    print(f"\n  推理时间 (warmup=10, runs=50)...")
    timing_pe = measure_inference_time(net_pe, pe_modules=pe_modules, device=device)
    print(f"  推理时间: {timing_pe['mean']:.1f} ± {timing_pe['std']:.1f} ms")

    # 分段计时
    seg_pe = measure_segmented_time(net_pe, pe_modules=pe_modules, device=device)
    print(f"  分段: encoder={seg_pe['encoder']:.1f}ms, "
          f"PE={seg_pe['pe']:.1f}ms, decoder={seg_pe['decoder']:.1f}ms")

    del net_pe, pe_modules
    torch.cuda.empty_cache()

    # ============================================
    # C. 论文表格
    # ============================================
    overhead_ms = timing_pe['mean'] - timing_base['mean']
    overhead_pct = overhead_ms / timing_base['mean'] * 100
    flops_overhead_pct = flops_pe_total / enc_flops_val * 100

    print(f"\n\n{'=' * 75}")
    print("  表8: 模型效率对比")
    print(f"{'=' * 75}")
    print(f"  {'方法':<18} {'总参数':>8} {'新增参数':>10} {'FLOPs':>10} {'推理时间':>16}")
    print(f"  {'─' * 64}")
    print(f"  {'MedSAM2':<18} {total_base / 1e6:.2f}M{'':>5} {'—':>10} "
          f"{enc_flops_val / 1e9:.1f}G{'':>5} {timing_base['mean']:.1f}±{timing_base['std']:.1f}ms")
    print(f"  {'PE-MedSAM2':<18} {total_pe / 1e6:.2f}M{'':>5} "
          f"{pe_total / 1e3:.1f}K{'':>5} {total_flops / 1e9:.1f}G{'':>5} "
          f"{timing_pe['mean']:.1f}±{timing_pe['std']:.1f}ms")
    print(f"  {'─' * 64}")
    print(f"  开销: +{pe_total / 1e3:.1f}K参数(+{pe_total / total_base * 100:.2f}%), "
          f"+{flops_pe_total / 1e9:.2f}GFLOPs(+{flops_overhead_pct:.1f}%), "
          f"+{overhead_ms:.1f}ms(+{overhead_pct:.1f}%)")

    print(f"\n  时间分段对比:")
    print(f"    {'阶段':<15} {'MedSAM2':>10} {'PE-MedSAM2':>12} {'差值':>10}")
    print(f"    {'─' * 49}")
    print(f"    {'Encoder':<15} {seg_base['encoder']:>9.1f}ms {seg_pe['encoder']:>11.1f}ms "
          f"{seg_pe['encoder'] - seg_base['encoder']:>+9.1f}ms")
    if seg_pe['pe'] > 0:
        print(f"    {'PE 模块':<15} {'—':>10} {seg_pe['pe']:>11.1f}ms "
              f"{seg_pe['pe']:>+9.1f}ms")
    print(f"    {'Decoder':<15} {seg_base['decoder']:>9.1f}ms {seg_pe['decoder']:>11.1f}ms "
          f"{seg_pe['decoder'] - seg_base['decoder']:>+9.1f}ms")

    print(f"\n{'=' * 75}")
    print("  表9: PE-MedSAM2 模块参数与 FLOPs 分布")
    print(f"{'=' * 75}")
    print(f"  {'模块':<10} {'参数量':>10} {'参数占比':>8} {'FLOPs':>12} {'FLOPs占比':>10}")
    print(f"  {'─' * 52}")
    pe_flops_map = {
        'LRA': flops_lra,
        'PFFE': flops_pffe,
        'ULA': flops_ula,
        'DSA': flops_dsa,
    }
    for n, p in pe_detail.items():
        if n == '_total': continue
        ppct = (p / pe_total * 100) if pe_total > 0 else 0
        f_val = pe_flops_map.get(n, 0)
        fpct = (f_val / flops_pe_total * 100) if flops_pe_total > 0 else 0
        f_str = f"{f_val / 1e6:.1f}M" if f_val < 1e9 else f"{f_val / 1e9:.2f}G"
        print(f"  {n:<10} {p:>10,} {ppct:>7.1f}% {f_str:>12} {fpct:>9.1f}%")
    print(f"  {'─' * 52}")
    print(f"  {'总计':<10} {pe_total:>10,} {'100.0%':>8} {flops_pe_total / 1e9:.2f}G{'':>5} {'100.0%':>10}")
    print(f"{'=' * 75}")

    print(f"\n  GPU: {gpu_name}")
    print(f"  注: FLOPs 为理论值 (encoder 用 {'fvcore' if method == 'fvcore' else '手动估算'}, "
          f"PE 模块用手动计算)")


if __name__ == '__main__':
    main()