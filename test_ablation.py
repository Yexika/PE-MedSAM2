#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
test_ablation.py - PE-MedSAM2 消融实验测试脚本 (5 模块: LRA+PFFE+ULA+DSA+MAL)
==============================================================================
对应 run_ablation.py 的 ablation 目录

实验设计 (删除法: 从完整模型逐个去除):
  E0_Baseline:  MedSAM2 (无 PE)
  E1_Full:      LRA + PFFE + ULA + DSA + MAL (完整 PE)
  E2_wo_LRA:    Full - LRA
  E3_wo_PFFE:   Full - PFFE (DSA 通过 fallback boundary_map 正常运行)
  E4_wo_ULA:    Full - ULA
  E5_wo_DSA:    Full - DSA
  E6_wo_MAL:    Full 但用标准 Dice+BCE loss (无边界损失)

指标: DSC, IoU, ASD

用法:
  python test_ablation.py                    # 测试全部
  python test_ablation.py --exp E2_wo_LRA    # 只测试指定实验
  python test_ablation.py --exp E1_Full E5_wo_DSA  # 测试多个
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
from collections import OrderedDict

BASE_PATH = '/root/autodl-tmp/PE-MedSAM2'
sys.path.insert(0, BASE_PATH)

import cfg_pe as cfg
from func_2d.utils import get_network
from func_2d.dataset import REFUGE
from func_2d.dataset_modified import MultiDataset
from func_2d.pe_utils import create_pe_modules, apply_pe_to_features
from func_2d.filter_utils import filter_abnormal_prediction, AbnormalStats

SAVE_ROOT = os.path.join(BASE_PATH, 'Save_2D', 'ablation')
WEIGHT_PATH = os.path.join(BASE_PATH, 'weight')
DATA_ROOT = '/root/autodl-tmp/datasets'

# ★ 与论文一致: CVC-ClinicDB + ISIC17
DATASETS = ['CVC-ClinicDB', 'ISIC17']

# ============== 实验配置 (5 模块) ==============
EXPERIMENTS = OrderedDict([
    ('E0_Baseline', {
        'weight_pattern': 'Medsam2_{dataset}.pth',
        'weight_dir': '/root/autodl-tmp/Medical-SAM2-main/weight',
        'description': 'MedSAM2 Baseline',
        'use_pe': False,
    }),
    ('E1_Full', {
        'weight_pattern': 'PE_MedSAM_{dataset}.pth',
        'weight_dir': os.path.join(BASE_PATH, 'weight'),
        'description': 'Full PE',
        'use_pe': True,
        'pe_flags': {'use_lra': True, 'use_pffe': True,
                     'use_ula': True, 'use_dsa': True},
    }),
    ('E2_wo_LRA', {
        'weight_pattern': 'PE_MedSAM_{dataset}.pth',
        'weight_dir': os.path.join(WEIGHT_PATH, 'ablation', 'E2_wo_LRA'),
        'description': 'w/o LRA',
        'use_pe': True,
        'pe_flags': {'use_lra': False, 'use_pffe': True,
                     'use_ula': True, 'use_dsa': True},
    }),
    ('E3_wo_PFFE', {
        'weight_pattern': 'PE_MedSAM_{dataset}.pth',
        'weight_dir': os.path.join(WEIGHT_PATH, 'ablation', 'E3_wo_PFFE'),
        'description': 'w/o PFFE',
        'use_pe': True,
        'pe_flags': {'use_lra': True, 'use_pffe': False,
                     'use_ula': True, 'use_dsa': True},
    }),
    ('E4_wo_ULA', {
        'weight_pattern': 'PE_MedSAM_{dataset}.pth',
        'weight_dir': os.path.join(WEIGHT_PATH, 'ablation', 'E4_wo_ULA'),
        'description': 'w/o ULA',
        'use_pe': True,
        'pe_flags': {'use_lra': True, 'use_pffe': True,
                     'use_ula': False, 'use_dsa': True},
    }),
    ('E5_wo_DSA', {
        'weight_pattern': 'PE_MedSAM_{dataset}.pth',
        'weight_dir': os.path.join(WEIGHT_PATH, 'ablation', 'E5_wo_DSA'),
        'description': 'w/o DSA',
        'use_pe': True,
        'pe_flags': {'use_lra': True, 'use_pffe': True,
                     'use_ula': True, 'use_dsa': False},
    }),
    ('E6_wo_MAL', {
        'weight_pattern': 'PE_MedSAM_{dataset}.pth',
        'weight_dir': os.path.join(WEIGHT_PATH, 'ablation', 'E6_wo_MAL'),
        'description': 'w/o MAL',
        'use_pe': True,
        'pe_flags': {'use_lra': True, 'use_pffe': True,
                     'use_ula': True, 'use_dsa': True},
        # ★ E6 结构上与 E1 完全一致, 只是训练时 loss 不同 (lambda_boundary=0)
        #   测试时不需要特殊处理, 差异体现在训练后的权重中
    }),
])

# ============== 指标计算 ==============
from scipy import ndimage
from scipy.ndimage import distance_transform_edt


def compute_dice(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    return (2. * np.logical_and(pred, gt).sum()) / (pred.sum() + gt.sum() + 1e-6)


def compute_iou(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    return np.logical_and(pred, gt).sum() / (np.logical_or(pred, gt).sum() + 1e-6)


def compute_surface_distances(pred, gt, spacing=(1.0, 1.0)):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if pred.sum() == 0 or gt.sum() == 0:
        return None, None
    pred_border = pred ^ ndimage.binary_erosion(pred)
    gt_border = gt ^ ndimage.binary_erosion(gt)
    if pred_border.sum() == 0 or gt_border.sum() == 0:
        return None, None
    dt_gt = distance_transform_edt(~gt, sampling=spacing)
    dt_pred = distance_transform_edt(~pred, sampling=spacing)
    return dt_gt[pred_border], dt_pred[gt_border]


def compute_asd(pred, gt, spacing=(1.0, 1.0)):
    d_p2g, d_g2p = compute_surface_distances(pred, gt, spacing)
    if d_p2g is None:
        return float('inf')
    return (np.mean(d_p2g) + np.mean(d_g2p)) / 2


def get_feat_sizes(image_size):
    return [(image_size // 4, image_size // 4),
            (image_size // 8, image_size // 8),
            (image_size // 16, image_size // 16)]


# ============== 核心测试函数 ==============
def test_model(args, net, test_loader, dataset_name, pe_modules=None):
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    net.eval()
    if pe_modules is not None:
        for module in pe_modules.values():
            if isinstance(module, nn.Module):
                module.eval()

    device = torch.device('cuda', args.gpu_device)
    feat_sizes = get_feat_sizes(args.image_size)
    embed_size = args.image_size // 16
    memory_bank_list = []

    results = []
    total_intersection, total_union, total_pred_sum, total_gt_sum = 0, 0, 0, 0
    all_pred_to_gt, all_gt_to_pred = [], []

    abnormal_stats = AbnormalStats()
    area_threshold = getattr(args, 'filter_area_threshold', 0.9)
    min_dice_threshold = getattr(args, 'filter_min_dice', 0.50)

    pe_tag = "PE" if pe_modules else "SAM2"
    with torch.no_grad():
        for ind, pack in enumerate(tqdm(test_loader, desc=f'测试 {dataset_name} [{pe_tag}]', leave=False)):
            to_cat_memory, to_cat_memory_pos, to_cat_image_embed = [], [], []

            name = pack['image_meta_dict']['filename_or_obj']
            imgs = pack['image'].to(dtype=torch.float32, device=device)
            masks = pack['mask'].to(dtype=torch.float32, device=device)

            if 'pt' in pack:
                pt = pack['pt'].to(device=device).unsqueeze(1)
                point_labels = pack['p_label'].to(device=device).unsqueeze(1)
                coords_torch, labels_torch = pt.float(), point_labels.int()
            else:
                coords_torch, labels_torch = None, None

            backbone_out = net.forward_image(imgs)
            _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
            B = vision_feats[-1].size(1)

            if len(memory_bank_list) == 0:
                vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(
                    torch.zeros(1, B, net.hidden_dim)).to(device="cuda")
                vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(
                    torch.zeros(1, B, net.hidden_dim)).to(device="cuda")
            else:
                for el in memory_bank_list:
                    to_cat_memory.append(el[0].cuda(non_blocking=True).flatten(2).permute(2, 0, 1))
                    to_cat_memory_pos.append(el[1].cuda(non_blocking=True).flatten(2).permute(2, 0, 1))
                    to_cat_image_embed.append(el[3].cuda(non_blocking=True))
                memory_stack = torch.stack(to_cat_memory, dim=0)
                memory_pos_stack = torch.stack(to_cat_memory_pos, dim=0)
                image_embed_stack = torch.stack(to_cat_image_embed, dim=0)
                vf_temp = vision_feats[-1].permute(1, 0, 2).view(B, -1, embed_size, embed_size).reshape(B, -1)
                image_embed_stack = F.normalize(image_embed_stack, p=2, dim=1)
                vf_temp = F.normalize(vf_temp, p=2, dim=1)
                sim = F.softmax(torch.mm(image_embed_stack, vf_temp.t()).t(), dim=1)
                idx = torch.multinomial(sim, num_samples=B, replacement=True).squeeze(1)
                mem_new = memory_stack[idx].squeeze(3).permute(1, 2, 0, 3)
                memory = mem_new.reshape(-1, mem_new.size(2), mem_new.size(3))
                mem_pos_new = memory_pos_stack[idx].squeeze(3).permute(1, 2, 0, 3)
                memory_pos = mem_pos_new.reshape(-1, mem_new.size(2), mem_new.size(3))
                vision_feats[-1] = net.memory_attention(
                    curr=[vision_feats[-1]], curr_pos=[vision_pos_embeds[-1]],
                    memory=memory, memory_pos=memory_pos, num_obj_ptr_tokens=0)

            feats = [feat.permute(1, 2, 0).view(B, -1, *fs)
                     for feat, fs in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
            image_embed = feats[-1]

            if pe_modules is not None:
                image_embed, _, _ = apply_pe_to_features(image_embed, pe_modules)

            high_res_feats = feats[:-1]

            points = (coords_torch, labels_torch) if coords_torch is not None else None
            flag = True
            se, de = net.sam_prompt_encoder(points=points, boxes=None, masks=None, batch_size=B)
            low_res_masks, iou_pred, _, _ = net.sam_mask_decoder(
                image_embeddings=image_embed, image_pe=net.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se, dense_prompt_embeddings=de,
                multimask_output=False, repeat_image=False, high_res_features=high_res_feats)

            pred = F.interpolate(low_res_masks, size=(args.out_size, args.out_size))

            high_res_masks = F.interpolate(low_res_masks, size=(args.image_size, args.image_size),
                                           mode="bilinear", align_corners=False)
            maskmem_features, maskmem_pos_enc = net._encode_new_memory(
                current_vision_feats=vision_feats, feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks, is_mask_from_pts=flag)
            maskmem_features = maskmem_features.to(torch.bfloat16).to(device=device, non_blocking=True)
            maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16).to(device=device, non_blocking=True)

            if len(memory_bank_list) < 16:
                for b in range(maskmem_features.size(0)):
                    memory_bank_list.append([
                        maskmem_features[b].unsqueeze(0), maskmem_pos_enc[b].unsqueeze(0),
                        iou_pred[b, 0], image_embed[b].reshape(-1).detach()])

            pred_binary = (pred > 0).float()

            for b in range(B):
                sample_name = name[b] if isinstance(name, (list, tuple)) else name
                sample_name = os.path.basename(str(sample_name)).replace('.png', '').replace('.jpg', '')

                pred_np = pred_binary[b, 0].cpu().numpy()
                gt_np = (masks[b, 0].cpu().numpy() > 0.5).astype(np.uint8)

                filtered_pred, is_abnormal, reason = filter_abnormal_prediction(
                    pred_np, gt_np, area_threshold, min_dice_threshold)

                abnormal_stats.update(sample_name, is_abnormal, reason)

                if is_abnormal:
                    continue

                dice = compute_dice(pred_np, gt_np)
                iou_val = compute_iou(pred_np, gt_np)
                asd = compute_asd(pred_np, gt_np)

                results.append({
                    'name': sample_name,
                    'dice': dice, 'iou': iou_val,
                    'asd': asd if not np.isinf(asd) else -1
                })

                pf = pred_np.flatten().astype(bool)
                gf = gt_np.flatten().astype(bool)
                total_intersection += np.sum(pf & gf)
                total_union += np.sum(pf | gf)
                total_pred_sum += np.sum(pf)
                total_gt_sum += np.sum(gf)

                d_p2g, d_g2p = compute_surface_distances(pred_np, gt_np)
                if d_p2g is not None:
                    all_pred_to_gt.extend(d_p2g.tolist())
                    all_gt_to_pred.extend(d_g2p.tolist())

    if len(results) == 0:
        print(f"  ⚠️  {dataset_name}: 所有样本都被过滤!")
        return None

    smooth = 1e-6
    global_dice = (2 * total_intersection + smooth) / (total_pred_sum + total_gt_sum + smooth)
    global_iou = (total_intersection + smooth) / (total_union + smooth)
    global_asd = (np.mean(all_pred_to_gt) + np.mean(all_gt_to_pred)) / 2 if all_pred_to_gt else float('inf')

    valid_asd = [r['asd'] for r in results if r['asd'] >= 0]
    avg_dice = np.mean([r['dice'] for r in results])
    avg_iou = np.mean([r['iou'] for r in results])
    avg_asd = np.mean(valid_asd) if valid_asd else float('inf')

    if abnormal_stats.abnormal > 0:
        print(f"  [过滤] {abnormal_stats.summary()}")

    return {
        'avg_dice': avg_dice, 'avg_iou': avg_iou,
        'avg_asd': avg_asd,
        'global_dice': global_dice, 'global_iou': global_iou,
        'global_asd': global_asd,
        'num_samples': len(results),
        'num_total': abnormal_stats.total,
        'num_abnormal': abnormal_stats.abnormal,
        'per_sample': results,
    }


def load_and_test(args, exp_name, exp_config, dataset_name, data_path):
    device = torch.device('cuda', args.gpu_device)
    weight_file = exp_config['weight_pattern'].format(dataset=dataset_name)
    weight_path = os.path.join(exp_config['weight_dir'], weight_file)

    if not os.path.exists(weight_path):
        print(f"  ⚠️  权重不存在: {weight_path}，跳过")
        return None

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=device, distribution=args.distributed)
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        net.load_state_dict(checkpoint['model'])
        train_epoch = checkpoint.get('epoch', '?')
        train_metrics = checkpoint.get('metrics', {})
        print(f"  [SAM2] 加载权重: epoch={train_epoch}, 训练Dice={train_metrics.get('avg_dice', 0):.4f}")
    else:
        net.load_state_dict(checkpoint)
        print(f"  [SAM2] 加载权重 (无元数据)")

    pe_modules = None
    if exp_config.get('use_pe', False):
        pe_flags = exp_config.get('pe_flags', {})
        args_copy = argparse_copy(args)
        for flag_name in ['use_lra', 'use_pffe', 'use_ula', 'use_dsa']:
            setattr(args_copy, flag_name, pe_flags.get(flag_name, False))

        # ★ 确保关键超参与训练一致
        args_copy.lra_rank = 4
        args_copy.pffe_scales = [3, 5, 7]
        args_copy.ula_compression = 16
        args_copy.dsa_sparsity = 0.25

        pe_modules = create_pe_modules(net.hidden_dim, args_copy)

        if isinstance(checkpoint, dict) and 'pe_modules' in checkpoint:
            saved_pe = checkpoint['pe_modules']
            loaded, skipped = [], []
            for name, module in pe_modules.items():
                if name in saved_pe and isinstance(module, nn.Module):
                    try:
                        module.load_state_dict(saved_pe[name])
                        loaded.append(name)
                    except Exception as e:
                        skipped.append(f"{name}({e})")
                elif name not in saved_pe:
                    skipped.append(f"{name}(not in ckpt)")
            print(f"  [PE] 加载模块: {loaded}")
            if skipped:
                print(f"  [PE] 跳过模块: {skipped}")
        else:
            print(f"  ⚠️  checkpoint 中没有 pe_modules!")

        for name, module in pe_modules.items():
            if isinstance(module, nn.Module):
                pe_modules[name] = module.to(device)
                module.eval()
        print(f"  [PE] 模块已加载到 GPU, 模式: eval")
    else:
        print(f"  [SAM2] 纯 baseline 模式, 不使用 PE 模块")

    test_dataset = MultiDataset(args, data_path, mode='Test', prompt='click', seed=args.seed)
    test_loader = DataLoader(test_dataset, batch_size=args.b, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    metrics = test_model(args, net, test_loader, dataset_name, pe_modules=pe_modules)

    if metrics:
        save_dir = os.path.join(SAVE_ROOT, exp_name, dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        pd.DataFrame(metrics['per_sample']).to_csv(
            os.path.join(save_dir, 'per_sample_metrics.csv'), index=False)

    del net, pe_modules
    torch.cuda.empty_cache()
    return metrics


def argparse_copy(args):
    import argparse
    return argparse.Namespace(**vars(args))


# ============== 对比分析 ==============
def delta_str(val, ref, higher_better=True):
    diff = val - ref
    if abs(diff) < 1e-6:
        return "  --  "
    if higher_better:
        arrow = "↑" if diff > 0 else "↓"
    else:
        arrow = "↓" if diff < 0 else "↑"
    if abs(diff) >= 1.0:
        return f"{arrow}{diff:+.2f}"
    else:
        return f"{arrow}{diff:+.4f}"


# ★ 论文名称映射
PAPER_NAMES = {
    'E0_Baseline': 'MedSAM2 Baseline',
    'E1_Full': 'Full PE (LRA+PFFE+ULA+DSA+MAL)',
    'E2_wo_LRA': 'w/o LRA',
    'E3_wo_PFFE': 'w/o PFFE',
    'E4_wo_ULA': 'w/o ULA',
    'E5_wo_DSA': 'w/o DSA',
    'E6_wo_MAL': 'w/o MAL',
}


def print_comparison_table(df):
    e1_ref = {}
    for _, row in df[df['Experiment'] == 'E1_Full'].iterrows():
        e1_ref[row['Dataset']] = row

    for dataset in DATASETS:
        ref = e1_ref.get(dataset)
        if ref is None:
            continue

        print(f"\n{'=' * 110}")
        print(f"  数据集: {dataset}    (对比基准: E1 Full PE = LRA+PFFE+ULA+DSA+MAL)")
        print(f"{'=' * 110}")
        print(f"  {'实验':<16} {'说明':<22} {'Dice':>8}  {'Δ':>10}  {'IoU':>8}  {'Δ':>10}  "
              f"{'ASD':>8}  {'Δ':>10}")
        print(f"  {'-' * 104}")

        dataset_rows = df[df['Dataset'] == dataset].sort_values('Experiment')
        for _, row in dataset_rows.iterrows():
            exp = row['Experiment']
            desc = PAPER_NAMES.get(exp, row['Description'])
            dice, iou, asd = row['Dice'], row['IoU'], row['ASD']

            if exp == 'E1_Full':
                print(f"  {exp:<16} {desc:<22} {dice:>8.4f}  {'  ★基准':>10}  "
                      f"{iou:>8.4f}  {'  ★基准':>10}  "
                      f"{asd:>8.2f}  {'  ★基准':>10}")
            else:
                d_dice = delta_str(dice, ref['Dice'], higher_better=True)
                d_iou = delta_str(iou, ref['IoU'], higher_better=True)
                d_asd = delta_str(asd, ref['ASD'], higher_better=False)
                print(f"  {exp:<16} {desc:<22} {dice:>8.4f}  {d_dice:>10}  "
                      f"{iou:>8.4f}  {d_iou:>10}  "
                      f"{asd:>8.2f}  {d_asd:>10}")

        print(f"  {'-' * 104}")

    # 模块贡献度排名 (Dice)
    print(f"\n{'=' * 80}")
    print("  模块贡献度排名 (移除后 Dice 平均下降幅度，越大说明越重要)")
    print(f"{'=' * 80}")

    module_impact = {}
    ablation_exps = [e for e in df['Experiment'].unique()
                     if e.startswith('E') and e not in ('E0_Baseline', 'E1_Full')]

    for exp in ablation_exps:
        drops = []
        for dataset in DATASETS:
            ref = e1_ref.get(dataset)
            row = df[(df['Experiment'] == exp) & (df['Dataset'] == dataset)]
            if ref is not None and len(row) > 0:
                drops.append(ref['Dice'] - row.iloc[0]['Dice'])
        if drops:
            module_impact[exp] = np.mean(drops)

    sorted_impact = sorted(module_impact.items(), key=lambda x: x[1], reverse=True)
    for rank, (exp, drop) in enumerate(sorted_impact, 1):
        paper_name = PAPER_NAMES.get(exp, exp).replace('w/o ', '')
        bar_len = int(max(0, drop) * 200)
        bar = "█" * min(bar_len, 40)
        if drop > 0:
            print(f"  {rank}. {paper_name:<8} Dice 下降 {drop:+.4f}  {bar}")
        else:
            print(f"  {rank}. {paper_name:<8} Dice 变化 {drop:+.4f}  ⚠️ 移除后反而提升!")

    # 模块贡献度排名 (ASD)
    print(f"\n{'=' * 80}")
    print("  模块贡献度排名 - ASD (移除后 ASD 平均上升幅度，越大说明越重要)")
    print(f"{'=' * 80}")

    module_impact_asd = {}
    for exp in ablation_exps:
        increases = []
        for dataset in DATASETS:
            ref = e1_ref.get(dataset)
            row = df[(df['Experiment'] == exp) & (df['Dataset'] == dataset)]
            if ref is not None and len(row) > 0:
                increases.append(row.iloc[0]['ASD'] - ref['ASD'])
        if increases:
            module_impact_asd[exp] = np.mean(increases)

    sorted_impact_asd = sorted(module_impact_asd.items(), key=lambda x: x[1], reverse=True)
    for rank, (exp, increase) in enumerate(sorted_impact_asd, 1):
        paper_name = PAPER_NAMES.get(exp, exp).replace('w/o ', '')
        bar_len = int(max(0, increase) * 20)
        bar = "█" * min(bar_len, 40)
        if increase > 0:
            print(f"  {rank}. {paper_name:<8} ASD 上升 {increase:+.4f}  {bar}")
        else:
            print(f"  {rank}. {paper_name:<8} ASD 变化 {increase:+.4f}  ⚠️ 移除后反而改善!")

    # E0 vs E1 总提升
    print(f"\n{'=' * 100}")
    print("  PE 整体提升 (E1 Full vs E0 Baseline)")
    print(f"{'=' * 100}")
    for dataset in DATASETS:
        e0_row = df[(df['Experiment'] == 'E0_Baseline') & (df['Dataset'] == dataset)]
        e1_row = df[(df['Experiment'] == 'E1_Full') & (df['Dataset'] == dataset)]
        if len(e0_row) > 0 and len(e1_row) > 0:
            e0, e1 = e0_row.iloc[0], e1_row.iloc[0]
            print(f"  {dataset:<12}  Dice: {e0['Dice']:.4f} → {e1['Dice']:.4f} ({e1['Dice']-e0['Dice']:+.4f})  "
                  f"IoU: {e0['IoU']:.4f} → {e1['IoU']:.4f} ({e1['IoU']-e0['IoU']:+.4f})")
            print(f"  {'':<12}  ASD:  {e0['ASD']:.2f} → {e1['ASD']:.2f} ({e1['ASD']-e0['ASD']:+.2f})")
    print(f"{'=' * 100}")

    # ★ 输出论文可用的 LaTeX 表格
    print_latex_table(df)


def print_latex_table(df):
    """输出可直接粘贴到论文的 LaTeX 消融表"""
    e1_ref = {}
    for _, row in df[df['Experiment'] == 'E1_Full'].iterrows():
        e1_ref[row['Dataset']] = row

    print(f"\n{'=' * 80}")
    print("  LaTeX 消融表 (可直接粘贴到论文)")
    print(f"{'=' * 80}")

    exp_order = ['E0_Baseline', 'E1_Full', 'E2_wo_LRA', 'E3_wo_PFFE',
                 'E4_wo_ULA', 'E5_wo_DSA', 'E6_wo_MAL']
    latex_labels = {
        'E0_Baseline': 'E0 & MedSAM2 Baseline',
        'E1_Full': 'E1 & Full PE',
        'E2_wo_LRA': 'E2 & w/o LRA',
        'E3_wo_PFFE': 'E3 & w/o PFFE',
        'E4_wo_ULA': 'E4 & w/o ULA',
        'E5_wo_DSA': 'E5 & w/o DSA',
        'E6_wo_MAL': 'E6 & w/o MAL',
    }

    for exp in exp_order:
        parts = [latex_labels.get(exp, exp)]
        for dataset in DATASETS:
            row = df[(df['Experiment'] == exp) & (df['Dataset'] == dataset)]
            ref = e1_ref.get(dataset)
            if len(row) == 0:
                parts.append('[TODO] & [TODO] & [TODO]')
                continue
            r = row.iloc[0]

            if exp == 'E1_Full':
                parts.append(f"{r['Dice']:.4f} (Ref.) & {r['IoU']:.4f} (Ref.) & {r['ASD']:.2f} (Ref.)")
            elif exp == 'E0_Baseline':
                parts.append(f"{r['Dice']:.4f} & {r['IoU']:.4f} & {r['ASD']:.2f}")
            else:
                d_dice = r['Dice'] - ref['Dice']
                d_iou = r['IoU'] - ref['IoU']
                d_asd = r['ASD'] - ref['ASD']
                parts.append(f"{r['Dice']:.4f} ({d_dice:+.4f}) & "
                             f"{r['IoU']:.4f} ({d_iou:+.4f}) & "
                             f"{r['ASD']:.2f} ({d_asd:+.2f})")

        print('    ' + ' & '.join(parts) + ' \\\\')


def main():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--exp', nargs='*', default=None)
    extra_args, _ = parser.parse_known_args()
    args = cfg.parse_args()

    # ★ 确保测试所需的默认参数存在
    defaults = {
        'use_pe': True, 'use_lra': True, 'use_pffe': True,
        'use_ula': True, 'use_dsa': True,
        'lra_rank': 4, 'pffe_scales': [3, 5, 7],
        'ula_compression': 16, 'dsa_sparsity': 0.25,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    if extra_args.exp:
        exps_to_test = OrderedDict(
            (k, v) for k, v in EXPERIMENTS.items() if k in extra_args.exp)
    else:
        exps_to_test = EXPERIMENTS

    if not exps_to_test:
        print(f"错误: 未找到指定实验。可用: {list(EXPERIMENTS.keys())}")
        return

    print(f"\n{'=' * 70}")
    print(f"PE-MedSAM2 消融实验测试 (LRA+PFFE+ULA+DSA+MAL)")
    print(f"共 {len(exps_to_test)} 个实验 × {len(DATASETS)} 个数据集")
    print(f"实验: {list(exps_to_test.keys())}")
    print(f"数据集: {DATASETS}")
    print(f"指标: DSC, IoU, ASD")
    print(f"{'=' * 70}\n")

    all_results = []

    for exp_name, exp_config in exps_to_test.items():
        print(f"\n{'#' * 70}")
        print(f"# {exp_name}: {PAPER_NAMES.get(exp_name, exp_config['description'])}")
        print(f"# PE: {'Yes' if exp_config.get('use_pe', False) else 'No (Baseline)'}")
        if exp_config.get('pe_flags'):
            name_map = {'use_lra': 'LRA', 'use_pffe': 'PFFE',
                        'use_ula': 'ULA', 'use_dsa': 'DSA'}
            active = [name_map[k] for k, v in exp_config['pe_flags'].items() if v]
            inactive = [name_map[k] for k, v in exp_config['pe_flags'].items() if not v]
            print(f"# 启用模块: {active}")
            if inactive:
                print(f"# 禁用模块: {inactive}")
        print(f"{'#' * 70}")

        for dataset_name in DATASETS:
            data_path = os.path.join(DATA_ROOT, dataset_name)
            if not os.path.exists(data_path):
                print(f"  数据集不存在: {data_path}，跳过")
                continue

            print(f"\n  📊 {exp_name} @ {dataset_name}")
            try:
                metrics = load_and_test(args, exp_name, exp_config, dataset_name, data_path)
                if metrics:
                    all_results.append({
                        'Experiment': exp_name,
                        'Description': PAPER_NAMES.get(exp_name, exp_config['description']),
                        'Dataset': dataset_name,
                        'Dice': metrics['avg_dice'],
                        'IoU': metrics['avg_iou'],
                        'ASD': metrics['avg_asd'],
                        'G_Dice': metrics['global_dice'],
                        'G_IoU': metrics['global_iou'],
                        'G_ASD': metrics['global_asd'],
                        'Samples': metrics['num_samples'],
                        'Filtered': metrics['num_abnormal'],
                    })
                    print(f"  ✅ Dice: {metrics['avg_dice']:.4f} | IoU: {metrics['avg_iou']:.4f} | "
                          f"ASD: {metrics['avg_asd']:.2f} | "
                          f"Samples: {metrics['num_samples']}/{metrics['num_total']}")
            except Exception as e:
                print(f"  ❌ 出错: {e}")
                import traceback; traceback.print_exc()

    if not all_results:
        print("\n⚠️  没有测试结果!")
        return

    df = pd.DataFrame(all_results)
    os.makedirs(SAVE_ROOT, exist_ok=True)

    csv_path = os.path.join(SAVE_ROOT, 'ablation_summary.csv')
    df.to_csv(csv_path, index=False)

    xlsx_path = os.path.join(SAVE_ROOT, 'ablation_summary.xlsx')
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Full Results', index=False)
            for metric in ['Dice', 'IoU', 'ASD']:
                pivot = df.pivot_table(
                    index=['Experiment', 'Description'],
                    columns='Dataset', values=metric
                ).reset_index()
                pivot.to_excel(writer, sheet_name=f'{metric}_Table', index=False)

            # vs E1_Full 对比表
            e1_data = df[df['Experiment'] == 'E1_Full'].set_index('Dataset')
            diff_rows = []
            for _, row in df.iterrows():
                if row['Dataset'] in e1_data.index:
                    ref = e1_data.loc[row['Dataset']]
                    diff_rows.append({
                        'Experiment': row['Experiment'],
                        'Description': row['Description'],
                        'Dataset': row['Dataset'],
                        'Dice': row['Dice'],
                        'Dice_Δ': row['Dice'] - ref['Dice'],
                        'IoU': row['IoU'],
                        'IoU_Δ': row['IoU'] - ref['IoU'],
                        'ASD': row['ASD'],
                        'ASD_Δ': row['ASD'] - ref['ASD'],
                    })
            if diff_rows:
                pd.DataFrame(diff_rows).to_excel(writer, sheet_name='vs_E1_Full', index=False)

        print(f"\n📊 Excel 已保存: {xlsx_path}")
    except Exception as e:
        print(f"Excel 保存失败 ({e}), CSV 仍可用")

    print_comparison_table(df)

    print(f"\n📁 结果文件:")
    print(f"   CSV:   {csv_path}")
    print(f"   Excel: {xlsx_path}")
    print(f"   逐样本: {SAVE_ROOT}/{{exp}}/{{dataset}}/per_sample_metrics.csv")


if __name__ == '__main__':
    main()