#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
run_ablation.py - PE-MedSAM2 消融实验自动化 (删除法)
======================================================
消融设计 (删除法: 从完整模型逐个去除):
  E0_Baseline:  MedSAM2 (复用已有权重，不训练)
  E1_Full:      LRA + PFFE + ULA + DSA + MAL (完整 PE)
  E2_wo_LRA:    Full - LRA
  E3_wo_PFFE:   Full - PFFE (DSA 通过 fallback boundary_map 正常运行)
  E4_wo_ULA:    Full - ULA
  E5_wo_DSA:    Full - DSA
  E6_wo_MAL:    Full 但用标准 Dice+BCE loss (lambda_boundary=0)

指标: DSC, IoU, ASD
"""

import os
import sys
import gc
import torch
from collections import OrderedDict

BASE_PATH = '/root/autodl-tmp/PE-MedSAM2'
sys.path.insert(0, BASE_PATH)

import cfg_pe as cfg
from train_pe_2d import train_single_dataset

WEIGHT_DIR = os.path.join(BASE_PATH, 'weight', 'ablation')
DATA_ROOT = '/root/autodl-tmp/datasets'

# ★ 与论文一致: CVC-ClinicDB + ISIC17
DATASETS = ['CVC-ClinicDB', 'ISIC17']


# ============== 消融配置 ==============
FULL_CONFIG = {
    'use_pe': True,
    'use_lra': True, 'use_pffe': True,
    'use_ula': True, 'use_dsa': True,
    'lambda_boundary': 0.3,
}

ABLATION_CONFIGS = OrderedDict([
    # 逐个去除
    ('E2_wo_LRA', {
        **FULL_CONFIG,
        'use_lra': False,
    }),
    ('E3_wo_PFFE', {
        **FULL_CONFIG,
        'use_pffe': False,
        # ★ DSA 通过 pe_modules.py 中的 fallback 正常运行
    }),
    ('E4_wo_ULA', {
        **FULL_CONFIG,
        'use_ula': False,
    }),
    ('E5_wo_DSA', {
        **FULL_CONFIG,
        'use_dsa': False,
    }),
    # 去除 MAL (用标准 Dice+BCE loss)
    ('E6_wo_MAL', {
        **FULL_CONFIG,
        'lambda_boundary': 0.0,
    }),
])


def main():
    args = cfg.parse_args()

    defaults = {
        'use_pe': True, 'use_lra': True, 'use_pffe': True,
        'use_ula': True, 'use_dsa': True,
        'lra_rank': 4, 'pffe_scales': [3, 5, 7],
        'ula_compression': 16, 'dsa_sparsity': 0.25,
        'lambda_dice': 1.0, 'lambda_bce': 0.5,
        'lambda_boundary': 0.3,
        'mal_warmup': 10,
        'warmup_epochs': 10, 'min_lr_ratio': 0.01,
        'max_grad_norm': 1.0, 'weight_decay': 0.01,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    args.early_stop_patience = 35

    total = len(ABLATION_CONFIGS) * len(DATASETS)
    done = 0

    for exp_name, config in ABLATION_CONFIGS.items():
        for dataset in DATASETS:
            done += 1

            weight_file = os.path.join(WEIGHT_DIR, exp_name, f'PE_MedSAM_{dataset}.pth')
            if os.path.exists(weight_file):
                print(f"\n[{done}/{total}] 跳过已存在: {weight_file}")
                continue

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # 显示本次实验信息
            print(f"\n{'#' * 70}")
            print(f"# [{done}/{total}] {exp_name} @ {dataset}")

            modules = ['lra', 'pffe', 'ula', 'dsa']
            enabled = [m.upper() for m in modules if config.get(f'use_{m}', False)]
            disabled = [m.upper() for m in modules if not config.get(f'use_{m}', False)]

            en_str = ' + '.join(enabled) if enabled else '(none)'
            print(f'# 启用: {en_str}')
            if disabled:
                print(f'# 禁用: {", ".join(disabled)}')

            # 提示 PFFE fallback
            if not config.get('use_pffe', True) and config.get('use_dsa', True):
                print(f'# 注意: PFFE 已禁用, DSA 将使用 fallback boundary_map')

            lb = config.get('lambda_boundary', 0)
            if lb > 0:
                print(f'# MAL: λ_boundary={lb}')
            else:
                print('# MAL: 关闭 (标准 Dice+BCE)')

            print(f"# 早停: {args.early_stop_patience} epochs 无提升则停止")
            print(f"{'#' * 70}\n")

            for k, v in config.items():
                setattr(args, k, v)

            args.dataset = dataset
            args.data_path = os.path.join(DATA_ROOT, dataset)
            args.exp_name = f'ablation_{exp_name}_{dataset}'
            args.ablation_name = exp_name

            if not os.path.exists(args.data_path):
                print(f"数据集路径不存在: {args.data_path}，跳过")
                continue

            try:
                best_metrics = train_single_dataset(args, dataset, args.data_path)

                if best_metrics and os.path.exists(weight_file):
                    print(f"\n✅ 权重已保存: {weight_file}")
                    print(f"   Dice: {best_metrics.get('avg_dice', 0):.4f}, "
                          f"IoU: {best_metrics.get('avg_iou', 0):.4f}, "
                          f"ASD: {best_metrics.get('avg_asd', 'inf')}")

            except Exception as e:
                print(f"\n❌ {exp_name} @ {dataset} 出错: {e}")
                import traceback
                traceback.print_exc()
            finally:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

    # 汇总
    print(f"\n{'=' * 70}")
    print("PE-MedSAM2 消融实验完成! 权重目录:")
    print(f"{'=' * 70}")
    if os.path.exists(WEIGHT_DIR):
        for exp_dir in sorted(os.listdir(WEIGHT_DIR)):
            exp_path = os.path.join(WEIGHT_DIR, exp_dir)
            if os.path.isdir(exp_path):
                files = [f for f in os.listdir(exp_path) if f.endswith('.pth')]
                for f in sorted(files):
                    size_mb = os.path.getsize(os.path.join(exp_path, f)) / 1024 / 1024
                    print(f"  {exp_dir}/{f}  ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()