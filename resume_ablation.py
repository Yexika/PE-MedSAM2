#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
resume_ablation.py - 从断点恢复消融实验训练
=============================================
用法:
  python resume_ablation.py

会自动:
  1. 加载已有权重 (含 model + optimizer + scheduler + pe_modules + epoch)
  2. 从保存的 epoch 继续训练到 100 epoch
  3. 保存到同一路径 (如果新 dice 更高才覆盖)
"""

import os
import sys
import time
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

BASE_PATH = '/root/autodl-tmp/PE-MedSAM2'
sys.path.insert(0, BASE_PATH)

import cfg_pe as cfg
from conf import settings
from func_2d.utils import get_network, set_log_dir, create_logger
from func_2d.dataset_modified import MultiDataset
from func_2d.pe_utils import create_pe_modules, create_mal_loss
import func_2d.function_pe as function

DATA_ROOT = '/root/autodl-tmp/datasets'

# ============== 配置: 修改这里 ==============
EXP_NAME = 'E6_wo_MAL'
DATASET = 'ISIC17'
WEIGHT_PATH = os.path.join(BASE_PATH, 'weight', 'ablation', EXP_NAME, f'PE_MedSAM_{DATASET}.pth')
TOTAL_EPOCHS = 100
EARLY_STOP_PATIENCE = 35

# E6_wo_MAL 的模块配置: 全开但 lambda_boundary=0
PE_FLAGS = {
    'use_lra': True, 'use_pffe': True,
    'use_ula': True, 'use_dsa': True,
}
LOSS_CONFIG = {
    'lambda_boundary': 0.0,  # ★ E6_wo_MAL: 无边界损失
}
# =============================================


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                    min_lr_ratio=0.01, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        return max(min_lr_ratio, cosine_decay)
    return LambdaLR(optimizer, lr_lambda)


def main():
    args = cfg.parse_args()

    # 设置参数
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

    # 应用实验特定配置
    for k, v in PE_FLAGS.items():
        setattr(args, k, v)
    for k, v in LOSS_CONFIG.items():
        setattr(args, k, v)

    args.dataset = DATASET
    args.data_path = os.path.join(DATA_ROOT, DATASET)
    args.exp_name = f'resume_{EXP_NAME}_{DATASET}'
    args.ablation_name = EXP_NAME

    # ============== 检查权重 ==============
    if not os.path.exists(WEIGHT_PATH):
        print(f"❌ 权重不存在: {WEIGHT_PATH}")
        return

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    GPUdevice = torch.device('cuda', args.gpu_device)

    # ============== 加载模型 ==============
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)
    checkpoint = torch.load(WEIGHT_PATH, map_location=GPUdevice, weights_only=False)

    net.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint.get('epoch', 0) + 1  # 从下一个 epoch 开始
    best_metrics = checkpoint.get('metrics', {})
    best_dice = best_metrics.get('avg_dice', 0)

    print(f"\n{'=' * 70}")
    print(f"断点续训: {EXP_NAME} @ {DATASET}")
    print(f"{'=' * 70}")
    print(f"权重路径: {WEIGHT_PATH}")
    print(f"保存时 epoch: {checkpoint.get('epoch', '?')}")
    print(f"保存时 best Dice: {best_dice:.4f}")
    print(f"将从 epoch {start_epoch} 训练到 epoch {TOTAL_EPOCHS - 1}")
    print(f"剩余 epochs: {TOTAL_EPOCHS - start_epoch}")
    print(f"MAL: lambda_boundary={args.lambda_boundary}")
    print(f"{'=' * 70}\n")

    if start_epoch >= TOTAL_EPOCHS:
        print(f"⚠️  已经训练完成 ({start_epoch} >= {TOTAL_EPOCHS})，无需继续")
        return

    # ============== 初始化 PE 模块 ==============
    pe_modules = create_pe_modules(net.hidden_dim, args)

    # 加载 PE 权重
    if 'pe_modules' in checkpoint:
        for name, module in pe_modules.items():
            if name in checkpoint['pe_modules']:
                try:
                    module.load_state_dict(checkpoint['pe_modules'][name])
                    print(f"  [PE] 已恢复: {name}")
                except Exception as e:
                    print(f"  ⚠️  {name} 恢复失败: {e}")
    else:
        print("  ⚠️  checkpoint 中没有 pe_modules，将用随机初始化继续")

    for name, module in pe_modules.items():
        pe_modules[name] = module.to(GPUdevice)

    mal_loss = create_mal_loss(args, GPUdevice)
    args._pe_modules = pe_modules
    args._mal_loss = mal_loss

    # ============== 优化器 ==============
    base_params = [p for p in net.parameters() if p.requires_grad]
    pe_params = []
    for module in pe_modules.values():
        pe_params.extend([p for p in module.parameters() if p.requires_grad])

    param_groups = [{'params': base_params, 'lr': args.lr}]
    if pe_params:
        param_groups.append({'params': pe_params, 'lr': args.lr})

    warmup_epochs = getattr(args, 'warmup_epochs', 10)
    weight_decay = getattr(args, 'weight_decay', 0.01)
    max_grad_norm = getattr(args, 'max_grad_norm', 1.0)

    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)

    # ============== 数据集 ==============
    train_dataset = MultiDataset(args, args.data_path, mode='Training', prompt='click', seed=args.seed)
    test_dataset = MultiDataset(args, args.data_path, mode='Test', prompt='click', seed=args.seed)
    nice_train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=True, drop_last=True)
    nice_test_loader = DataLoader(test_dataset, batch_size=args.b, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)

    steps_per_epoch = len(nice_train_loader)
    total_training_steps = TOTAL_EPOCHS * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    # ============== 学习率调度器 ==============
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_training_steps,
        min_lr_ratio=getattr(args, 'min_lr_ratio', 0.01))

    # 恢复 optimizer 和 scheduler 状态
    if 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("  [Optimizer] 状态已恢复")
        except Exception as e:
            print(f"  ⚠️  Optimizer 恢复失败 ({e})，将用新 optimizer 继续")

    if 'scheduler' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("  [Scheduler] 状态已恢复")
        except Exception as e:
            print(f"  ⚠️  Scheduler 恢复失败 ({e})，手动快进...")
            # 手动快进 scheduler 到正确位置
            for _ in range(start_epoch * steps_per_epoch):
                scheduler.step()
            print(f"  [Scheduler] 已快进到 step {start_epoch * steps_per_epoch}")
    else:
        # 手动快进
        for _ in range(start_epoch * steps_per_epoch):
            scheduler.step()
        print(f"  [Scheduler] 已快进到 step {start_epoch * steps_per_epoch}")

    # 当前学习率
    current_lr = optimizer.param_groups[0]['lr']
    print(f"  [LR] 当前学习率: {current_lr:.2e}")

    # ============== 日志 ==============
    log_path = os.path.join('logs', f'resume_{EXP_NAME}_{DATASET}')
    args.path_helper = set_log_dir(log_path, args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.handlers = logger.handlers[-1:]

    log_dir = os.path.join(settings.LOG_DIR, args.net,
                           f'resume_{EXP_NAME}_{DATASET}_{settings.TIME_NOW}')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    logger.info(f"断点续训: {EXP_NAME} @ {DATASET}, 从 epoch {start_epoch} 到 {TOTAL_EPOCHS}")

    # ============== 训练循环 ==============
    global_step = start_epoch * steps_per_epoch
    no_improve_count = 0

    for epoch in range(start_epoch, TOTAL_EPOCHS):
        # 训练
        net.train()
        for module in pe_modules.values():
            module.train()

        time_start = time.time()
        loss, global_step = function.train_sam_with_scheduler(
            args, net, optimizer, nice_train_loader, epoch, writer,
            scheduler=scheduler, max_grad_norm=max_grad_norm, global_step=global_step)

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate', current_lr, epoch)
        logger.info(f'训练损失: {loss:.4f} @ epoch {epoch}, lr: {current_lr:.2e}')
        print(f'Epoch {epoch}: loss={loss:.4f}, lr={current_lr:.2e}, 耗时 {time.time()-time_start:.1f}s')

        # 验证
        net.eval()
        for module in pe_modules.values():
            module.eval()

        if epoch % args.val_freq == 0 or epoch == TOTAL_EPOCHS - 1:
            tol, metrics_tuple = function.validation_sam(args, nice_test_loader, epoch, net)
            avg_iou, avg_dice, global_dice, global_iou, avg_hd95, avg_asd, global_hd95, global_asd = metrics_tuple

            logger.info(f'[Epoch {epoch}] Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}, ASD: {avg_asd:.2f}')

            writer.add_scalar('val/avg_dice', avg_dice, epoch)
            writer.add_scalar('val/avg_iou', avg_iou, epoch)

            if avg_dice > best_dice:
                best_dice = avg_dice
                best_metrics = {
                    'epoch': epoch, 'avg_dice': avg_dice, 'avg_iou': avg_iou,
                    'avg_hd95': avg_hd95, 'avg_asd': avg_asd,
                    'global_dice': global_dice, 'global_iou': global_iou,
                    'global_hd95': global_hd95, 'global_asd': global_asd
                }

                save_dict = {
                    'model': net.state_dict(),
                    'epoch': epoch,
                    'metrics': best_metrics,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'pe_modules': {
                        name: module.state_dict()
                        for name, module in pe_modules.items()
                    }
                }
                torch.save(save_dict, WEIGHT_PATH)
                logger.info(f'*** 新最佳! Dice: {best_dice:.4f} @ epoch {epoch} ***')
                print(f'  ★ 新最佳! Dice: {best_dice:.4f} 已保存')
                no_improve_count = 0
            else:
                no_improve_count += 1
                if EARLY_STOP_PATIENCE > 0 and no_improve_count >= EARLY_STOP_PATIENCE:
                    logger.info(f'⏹ 早停: {EARLY_STOP_PATIENCE} epochs 无提升')
                    print(f'  ⏹ 早停! 最佳 Dice: {best_dice:.4f} @ epoch {best_metrics.get("epoch")}')
                    break

    print(f"\n{'=' * 70}")
    print(f"续训完成! 最佳 Dice: {best_dice:.4f} @ epoch {best_metrics.get('epoch', '?')}")
    print(f"权重路径: {WEIGHT_PATH}")
    print(f"{'=' * 70}")

    writer.close()


if __name__ == '__main__':
    main()