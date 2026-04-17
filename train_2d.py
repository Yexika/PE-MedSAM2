#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
MedSAM2 2D 训练脚本
- 支持选择单个数据集或全部训练
- 权重保存到: /root/autodl-tmp/Medical-SAM2-main/weight/Medsam2_数据集名.pth
- 跟踪指标: Dice, IoU, HD95, ASD
"""

import os
import time
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import cfg_2d as cfg
import func_2d.function as function
from conf import settings
from func_2d.dataset import REFUGE
from func_2d.dataset_modified import MultiDataset
from func_2d.utils import get_network, set_log_dir, create_logger

# 基础路径
BASE_PATH = '/root/autodl-tmp/Medical-SAM2-main'
WEIGHT_PATH = os.path.join(BASE_PATH, 'weight')
DATA_ROOT = '/root/autodl-tmp/datasets'

# 支持的数据集列表
SUPPORTED_DATASETS = [
    #'BUSI',
    #'CVC-ClinicDB',
    'DSB18',
    #'ETIS-LaribPolypDB',
    #'ISIC17',
    #'ISIC18',
    'Kvasir-SEG'
]


def train_single_dataset(args, dataset_name, data_path):
    """训练单个数据集"""
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    GPUdevice = torch.device('cuda', args.gpu_device)
    os.makedirs(WEIGHT_PATH, exist_ok=True)
    weight_save_path = os.path.join(WEIGHT_PATH, f'Medsam2_{dataset_name}.pth')

    print("=" * 70)
    print(f"MedSAM2 2D 训练 - {dataset_name}")
    print("=" * 70)
    print(f"数据路径: {data_path}")
    print(f"图像尺寸: {args.image_size} x {args.image_size}")
    print(f"批量大小: {args.b}")
    print(f"学习率: {args.lr}")
    print(f"权重保存路径: {weight_save_path}")
    print("=" * 70)

    # 初始化网络
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

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

    # TensorBoard
    log_dir = os.path.join(settings.LOG_DIR, args.net, f'{dataset_name}_{settings.TIME_NOW}')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    best_dice = 0.0
    best_metrics = {}
    total_epochs = getattr(args, 'epochs', settings.EPOCH)

    for epoch in range(total_epochs):
        # 初始验证
        if epoch == 0:
            net.eval()
            tol, metrics_tuple = function.validation_sam(args, nice_test_loader, epoch, net)
            avg_iou, avg_dice, global_dice, global_iou, avg_hd95, avg_asd, global_hd95, global_asd = metrics_tuple
            logger.info(f'[Epoch {epoch}] 平均Dice: {avg_dice:.4f}, 平均IoU: {avg_iou:.4f}')

        # 训练
        net.train()
        time_start = time.time()
        loss = function.train_sam(args, net, optimizer, nice_train_loader, epoch, writer)
        logger.info(f'训练损失: {loss:.4f} @ epoch {epoch}')
        print(f'训练耗时: {time.time() - time_start:.2f}秒')

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
                torch.save({'model': net.state_dict(), 'epoch': epoch, 'metrics': best_metrics}, weight_save_path)
                logger.info(f'*** 最佳模型已保存! Dice: {best_dice:.4f} @ {weight_save_path} ***')

    logger.info("=" * 70)
    logger.info(f"训练完成! 最佳Epoch: {best_metrics.get('epoch', 'N/A')}, 最佳Dice: {best_metrics.get('avg_dice', 0):.4f}")
    logger.info("=" * 70)
    writer.close()

    return best_metrics


def main():
    args = cfg.parse_args()

    # 确定要训练的数据集
    if args.dataset.lower() == 'all':
        datasets_to_train = SUPPORTED_DATASETS
        print(f"\n将训练全部 {len(datasets_to_train)} 个数据集: {datasets_to_train}\n")
    else:
        datasets_to_train = [args.dataset]
        print(f"\n将训练单个数据集: {args.dataset}\n")

    all_results = {}

    for dataset_name in datasets_to_train:
        # 构建数据路径
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