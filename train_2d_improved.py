#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
MedSAM2 2D 训练脚本 (改进版)
- 学习率预热 (Warmup)
- 余弦退火 (Cosine Annealing)
- 梯度裁剪 (Gradient Clipping)
"""
#train_2d_improved.py
import os
import time
import math
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

import cfg_2d as cfg
import func_2d.function_improved as function
from conf import settings
from func_2d.dataset import REFUGE
from func_2d.dataset_modified import MultiDataset
from func_2d.utils import get_network, set_log_dir, create_logger

# 基础路径
BASE_PATH = '/root/autodl-tmp/Medical-SAM2-main'
WEIGHT_PATH = os.path.join(BASE_PATH, 'weight')
DATA_ROOT = '/root/autodl-tmp/datasets'

SUPPORTED_DATASETS = [
    'BUSI',
    'CVC-ClinicDB',
    'DSB18',
    'ETIS-LaribPolypDB',
    'ISIC17',
    'ISIC18',
    'Kvasir-SEG'
]


# ============== 学习率调度器 ==============

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, 
                                     min_lr_ratio=0.01, num_cycles=0.5):
    """
    带预热的余弦退火调度器
    
    Args:
        optimizer: 优化器
        num_warmup_steps: 预热步数
        num_training_steps: 总训练步数
        min_lr_ratio: 最小学习率比例 (相对于初始lr)
        num_cycles: 余弦周期数
    """
    def lr_lambda(current_step):
        # Warmup 阶段: 线性增加
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine 退火阶段
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        
        # 确保不低于最小学习率
        return max(min_lr_ratio, cosine_decay)
    
    return LambdaLR(optimizer, lr_lambda)


def train_single_dataset(args, dataset_name, data_path):
    """训练单个数据集"""
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    GPUdevice = torch.device('cuda', args.gpu_device)
    os.makedirs(WEIGHT_PATH, exist_ok=True)
    weight_save_path = os.path.join(WEIGHT_PATH, f'Medsam2_{dataset_name}.pth')

    # ============== 训练超参数 ==============
    # 可调参数
    warmup_epochs = getattr(args, 'warmup_epochs', 5)      # 预热 epoch 数
    min_lr_ratio = getattr(args, 'min_lr_ratio', 0.01)     # 最小学习率 = lr * min_lr_ratio
    max_grad_norm = getattr(args, 'max_grad_norm', 1.0)    # 梯度裁剪阈值
    weight_decay = getattr(args, 'weight_decay', 0.01)     # AdamW 权重衰减
    
    print("=" * 70)
    print(f"MedSAM2 2D 训练 (改进版) - {dataset_name}")
    print("=" * 70)
    print(f"数据路径: {data_path}")
    print(f"图像尺寸: {args.image_size} x {args.image_size}")
    print(f"批量大小: {args.b}")
    print(f"初始学习率: {args.lr}")
    print(f"预热 Epochs: {warmup_epochs}")
    print(f"最小学习率比例: {min_lr_ratio}")
    print(f"梯度裁剪阈值: {max_grad_norm}")
    print(f"权重衰减: {weight_decay}")
    print(f"权重保存路径: {weight_save_path}")
    print("=" * 70)

    # 初始化网络
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)
    
    # ★★★ 使用 AdamW 优化器 (带权重衰减) ★★★
    optimizer = optim.AdamW(
        net.parameters(), 
        lr=args.lr, 
        betas=(0.9, 0.999), 
        eps=1e-08, 
        weight_decay=weight_decay
    )

    # 日志设置
    log_path = os.path.join('logs', f'{args.exp_name}_{dataset_name}')
    args.path_helper = set_log_dir(log_path, args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(f"训练数据集: {dataset_name}")
    logger.info(args)

    # 数据变换
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    # 加载数据集
    if dataset_name == 'REFUGE':
        train_dataset = REFUGE(args, data_path, transform=transform_train, mode='Training')
        test_dataset = REFUGE(args, data_path, transform=transform_test, mode='Test')
    else:
        train_dataset = MultiDataset(args, data_path, mode='Training', prompt='click', seed=args.seed)
        test_dataset = MultiDataset(args, data_path, mode='Test', prompt='click', seed=args.seed)

    nice_train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=True, drop_last=True)
    nice_test_loader = DataLoader(test_dataset, batch_size=args.b, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)

    print(f"训练样本数: {len(train_dataset)}")
    print(f"测试样本数: {len(test_dataset)}")

    # ★★★ 计算总训练步数和预热步数 ★★★
    total_epochs = getattr(args, 'epochs', settings.EPOCH)
    steps_per_epoch = len(nice_train_loader)
    total_training_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    
    print(f"每 Epoch 步数: {steps_per_epoch}")
    print(f"总训练步数: {total_training_steps}")
    print(f"预热步数: {warmup_steps}")

    # ★★★ 创建学习率调度器 ★★★
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
        min_lr_ratio=min_lr_ratio
    )

    # TensorBoard
    log_dir = os.path.join(settings.LOG_DIR, args.net, f'{dataset_name}_{settings.TIME_NOW}')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    best_dice = 0.0
    best_metrics = {}
    global_step = 0

    for epoch in range(total_epochs):
        # 初始验证
        if epoch == 0:
            net.eval()
            tol, metrics_tuple = function.validation_sam(args, nice_test_loader, epoch, net)
            avg_iou, avg_dice, global_dice, global_iou, avg_hd95, avg_asd, global_hd95, global_asd = metrics_tuple
            logger.info(f'[Epoch {epoch}] 初始 - 平均Dice: {avg_dice:.4f}, 平均IoU: {avg_iou:.4f}')

        # ★★★ 训练 (带梯度裁剪) ★★★
        net.train()
        time_start = time.time()
        
        loss, global_step = function.train_sam_with_scheduler(
            args, net, optimizer, nice_train_loader, epoch, writer,
            scheduler=scheduler,
            max_grad_norm=max_grad_norm,
            global_step=global_step
        )
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate', current_lr, epoch)
        
        logger.info(f'训练损失: {loss:.4f} @ epoch {epoch}, lr: {current_lr:.2e}')
        print(f'训练耗时: {time.time() - time_start:.2f}秒, 当前学习率: {current_lr:.2e}')

        # 验证
        net.eval()
        if epoch % args.val_freq == 0 or epoch == total_epochs - 1:
            tol, metrics_tuple = function.validation_sam(args, nice_test_loader, epoch, net)
            avg_iou, avg_dice, global_dice, global_iou, avg_hd95, avg_asd, global_hd95, global_asd = metrics_tuple

            logger.info(f'[Epoch {epoch}] 平均Dice: {avg_dice:.4f}, 平均IoU: {avg_iou:.4f}, '
                        f'平均HD95: {avg_hd95:.2f}, 平均ASD: {avg_asd:.2f}')

            writer.add_scalar('val/avg_dice', avg_dice, epoch)
            writer.add_scalar('val/avg_iou', avg_iou, epoch)

            # 保存最佳模型
            if avg_dice > best_dice:
                best_dice = avg_dice
                best_metrics = {
                    'epoch': epoch, 'avg_dice': avg_dice, 'avg_iou': avg_iou,
                    'avg_hd95': avg_hd95, 'avg_asd': avg_asd,
                    'global_dice': global_dice, 'global_iou': global_iou,
                    'global_hd95': global_hd95, 'global_asd': global_asd
                }
                torch.save({
                    'model': net.state_dict(), 
                    'epoch': epoch, 
                    'metrics': best_metrics,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, weight_save_path)
                logger.info(f'*** 最佳模型已保存! Dice: {best_dice:.4f} @ {weight_save_path} ***')

    logger.info("=" * 70)
    logger.info(f"训练完成! 最佳Epoch: {best_metrics.get('epoch', 'N/A')}, 最佳Dice: {best_metrics.get('avg_dice', 0):.4f}")
    logger.info("=" * 70)
    writer.close()

    return best_metrics


def main():
    args = cfg.parse_args()
    
    # ★★★ 添加新的超参数默认值 ★★★
    if not hasattr(args, 'warmup_epochs'):
        args.warmup_epochs = 5
    if not hasattr(args, 'min_lr_ratio'):
        args.min_lr_ratio = 0.01
    if not hasattr(args, 'max_grad_norm'):
        args.max_grad_norm = 1.0
    if not hasattr(args, 'weight_decay'):
        args.weight_decay = 0.01

    # 确定要训练的数据集
    if args.dataset.lower() == 'all':
        datasets_to_train = SUPPORTED_DATASETS
        print(f"\n将训练全部 {len(datasets_to_train)} 个数据集: {datasets_to_train}\n")
    else:
        datasets_to_train = [args.dataset]
        print(f"\n将训练单个数据集: {args.dataset}\n")

    all_results = {}

    for dataset_name in datasets_to_train:
        if args.data_path and args.dataset.lower() != 'all':
            data_path = args.data_path
        else:
            data_path = os.path.join(DATA_ROOT, dataset_name)

        if not os.path.exists(data_path):
            print(f"警告: 数据集路径不存在，跳过 {dataset_name}: {data_path}")
            continue

        print(f"\n{'#' * 70}")
        print(f"# 开始训练: {dataset_name}")
        print(f"{'#' * 70}\n")

        try:
            best_metrics = train_single_dataset(args, dataset_name, data_path)
            all_results[dataset_name] = best_metrics
        except Exception as e:
            print(f"训练 {dataset_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 打印汇总结果
    print("\n" + "=" * 70)
    print("全部训练完成! 汇总结果:")
    print("=" * 70)
    for ds_name, metrics in all_results.items():
        print(f"{ds_name}: Dice={metrics.get('avg_dice', 0):.4f}, IoU={metrics.get('avg_iou', 0):.4f}, "
              f"HD95={metrics.get('avg_hd95', 'inf'):.2f}, ASD={metrics.get('avg_asd', 'inf'):.2f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
