#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
test_2d.py - PE 2D 测试脚本
============================
加载训练好的权重 + PE 模块权重, 测试并生成可视化结果。

关键点: 必须同时加载 PE 模块 (LRA+PFFE+ULA+DSA), 否则特征分布
       与训练时不匹配, 测试结果会显著下降。
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
import cv2
import openpyxl

BASE_PATH = '/root/autodl-tmp/PE-MedSAM2'
sys.path.insert(0, BASE_PATH)

from func_2d.filter_utils import filter_abnormal_prediction, AbnormalStats
from func_2d.pe_utils import create_pe_modules, apply_pe_to_features

import cfg_pe as cfg
from func_2d.utils import get_network
from func_2d.dataset import REFUGE
from func_2d.dataset_modified import MultiDataset

SAVE_ROOT = os.path.join(BASE_PATH, 'Save_2D')
WEIGHT_PATH = os.path.join(BASE_PATH, 'weight')
DATA_ROOT = '/root/autodl-tmp/datasets'
SUPPORTED_DATASETS = ['CVC-ClinicDB', 'ISIC17', 'ISIC18', 'Kvasir-SEG', 'DSB18']


# ============== 指标计算函数 ==============
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


def compute_hd95(pred, gt, spacing=(1.0, 1.0)):
    d_p2g, d_g2p = compute_surface_distances(pred, gt, spacing)
    if d_p2g is None:
        return float('inf')
    return np.percentile(np.concatenate([d_p2g, d_g2p]), 95)


def compute_asd(pred, gt, spacing=(1.0, 1.0)):
    d_p2g, d_g2p = compute_surface_distances(pred, gt, spacing)
    if d_p2g is None:
        return float('inf')
    return (np.mean(d_p2g) + np.mean(d_g2p)) / 2


# ============== 可视化函数 ==============
def overlay_mask(image, mask, alpha=0.5):
    color_mask = np.zeros_like(image)
    color_mask[:, :, 1] = mask * 255
    overlay = image.copy().astype(np.float32)
    overlay[mask > 0] = (1 - alpha) * image[mask > 0] + alpha * color_mask[mask > 0]
    return overlay.astype(np.uint8)


def overlay_contours(image, pred_mask, gt_mask):
    pred_uint8 = np.uint8(pred_mask * 255)
    gt_uint8 = np.uint8(gt_mask * 255)
    contours_pred, _ = cv2.findContours(pred_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_gt, _ = cv2.findContours(gt_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = image.copy()
    cv2.drawContours(overlay, contours_pred, -1, (0, 0, 255), 2)
    cv2.drawContours(overlay, contours_gt, -1, (255, 0, 0), 2)
    return overlay


def visualize_segmentation(image, pred_mask, gt_mask, background_color=(0, 0, 139)):
    overlay_image = np.zeros_like(image)
    overlay_image[:] = background_color
    false_positive = np.logical_and(pred_mask == 1, gt_mask == 0)
    false_negative = np.logical_and(pred_mask == 0, gt_mask == 1)
    overlay_image[gt_mask == 1] = [139, 0, 0]
    overlay_image[false_positive] = [0, 255, 0]
    overlay_image[false_negative] = [255, 255, 0]
    return overlay_image


def get_visualization(image_np, pred_mask, gt_mask, dataset_name):
    if dataset_name in ('CVC-ClinicDB', 'Kvasir-SEG', 'ETIS-LaribPolypDB', 'BUSI'):
        return overlay_mask(image_np, pred_mask)
    elif dataset_name in ('ISIC17', 'ISIC18'):
        return overlay_contours(image_np, pred_mask, gt_mask)
    elif dataset_name == 'DSB18':
        return visualize_segmentation(image_np, pred_mask, gt_mask)
    else:
        return overlay_mask(image_np, pred_mask)


def get_feat_sizes(image_size):
    return [(image_size // 4, image_size // 4),
            (image_size // 8, image_size // 8),
            (image_size // 16, image_size // 16)]


# ============== 测试函数 ==============
def test_model(args, net, test_loader, save_dir, dataset_name, pe_modules=None):
    """
    pe_modules: 训练好的 PE 模块字典, 测试时必须加载
    """
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    net.eval()

    # PE 模块设为 eval 模式
    if pe_modules is not None:
        for m in pe_modules.values():
            if isinstance(m, nn.Module):
                m.eval()
        print(f"  PE 模块已加载: {[n for n, m in pe_modules.items() if isinstance(m, nn.Module)]}")

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

    print(
        f"\n{'=' * 60}\n测试数据集: {dataset_name}\n测试样本数: {len(test_loader.dataset)}"
        f"\nPE 模块: {'✅ 已加载' if pe_modules else '❌ 未加载 (纯 MedSAM2)'}"
        f"\n保存路径: {save_dir}\n{'=' * 60}\n")

    with torch.no_grad():
        for ind, pack in enumerate(tqdm(test_loader, desc='测试中')):
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
                vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(
                    device="cuda")
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
                vision_feats[-1] = net.memory_attention(curr=[vision_feats[-1]], curr_pos=[vision_pos_embeds[-1]],
                                                        memory=memory, memory_pos=memory_pos, num_obj_ptr_tokens=0)

            feats = [feat.permute(1, 2, 0).view(B, -1, *fs) for feat, fs in zip(vision_feats[::-1], feat_sizes[::-1])][
                    ::-1]
            image_embed = feats[-1]

            # ★★★ 核心: 应用 PE 模块 (LRA → PFFE → ULA → DSA) ★★★
            if pe_modules is not None:
                image_embed, _, _ = apply_pe_to_features(image_embed, pe_modules)

            high_res_feats = feats[:-1]

            flag = True
            points = (coords_torch, labels_torch) if coords_torch is not None else None
            se, de = net.sam_prompt_encoder(points=points, boxes=None, masks=None, batch_size=B)
            low_res_masks, iou_pred, _, _ = net.sam_mask_decoder(
                image_embeddings=image_embed, image_pe=net.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se, dense_prompt_embeddings=de,
                multimask_output=False, repeat_image=False, high_res_features=high_res_feats)

            pred = F.interpolate(low_res_masks, size=(args.out_size, args.out_size))
            high_res_masks = F.interpolate(low_res_masks, size=(args.image_size, args.image_size), mode="bilinear",
                                           align_corners=False)

            maskmem_features, maskmem_pos_enc = net._encode_new_memory(
                current_vision_feats=vision_feats, feat_sizes=feat_sizes, pred_masks_high_res=high_res_masks,
                is_mask_from_pts=flag)
            maskmem_features = maskmem_features.to(torch.bfloat16).to(device=device, non_blocking=True)
            maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16).to(device=device, non_blocking=True)

            if len(memory_bank_list) < 16:
                for b in range(maskmem_features.size(0)):
                    memory_bank_list.append([maskmem_features[b].unsqueeze(0), maskmem_pos_enc[b].unsqueeze(0),
                                             iou_pred[b, 0], image_embed[b].reshape(-1).detach()])

            pred_sigmoid = torch.sigmoid(pred)

            for b in range(B):
                sample_name = name[b] if isinstance(name, (list, tuple)) else name
                sample_name = os.path.basename(str(sample_name)).replace('.png', '').replace('.jpg', '')
                pred_np = (pred_sigmoid[b, 0].cpu().numpy() > 0.5).astype(np.uint8)
                gt_np = (masks[b, 0].cpu().numpy() > 0.5).astype(np.uint8)

                filtered_pred, is_abnormal, reason = filter_abnormal_prediction(
                    pred_np, gt_np, area_threshold, min_dice_threshold
                )
                abnormal_stats.update(sample_name, is_abnormal, reason)

                if is_abnormal:
                    img_np = (imgs[b].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    vis_img = get_visualization(img_np, pred_np, gt_np, dataset_name)
                    cv2.imwrite(os.path.join(save_dir, f'ABNORMAL_{sample_name}.png'),
                                cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                    continue

                dice = compute_dice(pred_np, gt_np)
                iou_val = compute_iou(pred_np, gt_np)
                hd95 = compute_hd95(pred_np, gt_np)
                asd = compute_asd(pred_np, gt_np)

                results.append({
                    'name': sample_name,
                    'dice': dice,
                    'iou': iou_val,
                    'hd95': hd95 if not np.isinf(hd95) else -1,
                    'asd': asd if not np.isinf(asd) else -1,
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

                img_np = (imgs[b].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                vis_img = get_visualization(img_np, pred_np, gt_np, dataset_name)
                cv2.imwrite(os.path.join(save_dir, f'overlay_{sample_name}.png'),
                            cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

    smooth = 1e-6
    global_dice = (2 * total_intersection + smooth) / (total_pred_sum + total_gt_sum + smooth)
    global_iou = (total_intersection + smooth) / (total_union + smooth)
    global_hd95 = np.percentile(np.array(all_pred_to_gt + all_gt_to_pred), 95) if all_pred_to_gt else float('inf')
    global_asd = (np.mean(all_pred_to_gt) + np.mean(all_gt_to_pred)) / 2 if all_pred_to_gt else float('inf')

    valid_hd95 = [r['hd95'] for r in results if r['hd95'] >= 0]
    valid_asd = [r['asd'] for r in results if r['asd'] >= 0]
    avg_dice, avg_iou = np.mean([r['dice'] for r in results]), np.mean([r['iou'] for r in results])
    avg_hd95 = np.mean(valid_hd95) if valid_hd95 else float('inf')
    avg_asd = np.mean(valid_asd) if valid_asd else float('inf')

    print(f"\n[过滤统计] {abnormal_stats.summary()}")
    abnormal_stats.print_details()

    pd.DataFrame(results).to_csv(os.path.join(save_dir, 'per_sample_metrics.csv'), index=False)

    print(f"\n{'=' * 60}\n测试结果 - {dataset_name}\n{'=' * 60}")
    print(f"[平均指标] Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f} | HD95: {avg_hd95:.2f} | ASD: {avg_asd:.2f}")
    print(f"[全局指标] Dice: {global_dice:.4f} | IoU: {global_iou:.4f} | HD95: {global_hd95:.2f} | ASD: {global_asd:.2f}")
    print(f"{'=' * 60}\n可视化结果已保存到: {save_dir}\n")

    return {'dataset': dataset_name, 'num_samples': len(results),
            'avg_dice': avg_dice, 'avg_iou': avg_iou, 'avg_hd95': avg_hd95, 'avg_asd': avg_asd,
            'global_dice': global_dice, 'global_iou': global_iou, 'global_hd95': global_hd95,
            'global_asd': global_asd}


def test_single_dataset(args, dataset_name, data_path):
    """测试单个数据集 - 会自动加载 PE 模块"""
    device = torch.device('cuda', args.gpu_device)
    weight_path = os.path.join(WEIGHT_PATH, f'PE_MedSAM_{dataset_name}.pth')

    if not os.path.exists(weight_path):
        print(f"错误: 找不到权重文件 {weight_path}")
        return None

    save_dir = os.path.join(SAVE_ROOT, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'=' * 70}\nPE-MedSAM2 2D 测试 - {dataset_name}\n{'=' * 70}")
    print(f"数据路径: {data_path}\n权重路径: {weight_path}\n保存路径: {save_dir}\n{'=' * 70}\n")

    # 加载 SAM2 模型
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=device, distribution=args.distributed)
    checkpoint = torch.load(weight_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        net.load_state_dict(checkpoint['model'])
        print(f"加载 SAM2 权重成功! Epoch: {checkpoint.get('epoch', 'N/A')}")
        if 'metrics' in checkpoint:
            m = checkpoint['metrics']
            print(f"训练最佳指标 - Dice: {m.get('avg_dice', 0):.4f}, IoU: {m.get('avg_iou', 0):.4f}")
    else:
        net.load_state_dict(checkpoint)

    # ★★★ 创建并加载 PE 模块 ★★★
    pe_modules = None

    if isinstance(checkpoint, dict) and 'pe_modules' in checkpoint:
        print(f"\n🔧 检测到 PE 模块权重，正在加载...")

        # 创建 PE 模块 (使用训练时的配置)
        pe_modules = create_pe_modules(net.hidden_dim, args)

        # 加载训练好的权重
        loaded_count = 0
        for name, module in pe_modules.items():
            if isinstance(module, nn.Module) and name in checkpoint['pe_modules']:
                try:
                    module.load_state_dict(checkpoint['pe_modules'][name])
                    pe_modules[name] = module.to(device)
                    n_params = sum(p.numel() for p in module.parameters())
                    print(f"  ✅ {name.upper()}: 加载成功 ({n_params:,} 参数)")
                    loaded_count += 1
                except Exception as e:
                    print(f"  ❌ {name.upper()}: 加载失败 - {e}")
            elif isinstance(module, nn.Module):
                # 模块存在但 checkpoint 中没有权重 (如 PFFE 是无参数的)
                pe_modules[name] = module.to(device)
                n_params = sum(p.numel() for p in module.parameters())
                if n_params > 0:
                    print(f"  ⚠️ {name.upper()}: 无保存权重，使用初始化 ({n_params:,} 参数)")
                else:
                    print(f"  ✅ {name.upper()}: 无参数模块 (parameter-free)")

        print(f"  PE 模块加载完成: {loaded_count} 个模块\n")
    else:
        print(f"\n⚠️  权重文件中未找到 pe_modules，将以纯 MedSAM2 模式测试")
        print(f"    (如果这是 PE 训练的权重，测试结果会不准确!)\n")

    if dataset_name == 'REFUGE':
        test_dataset = REFUGE(args, data_path, transform=transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()]), mode='Test')
    else:
        test_dataset = MultiDataset(args, data_path, mode='Test', prompt='click', seed=args.seed)

    test_loader = DataLoader(test_dataset, batch_size=args.b, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    return test_model(args, net, test_loader, save_dir, dataset_name, pe_modules=pe_modules)


def main():
    args = cfg.parse_args()

    # 添加 PE 参数默认值 (用于 create_pe_modules)
    pe_defaults = {
        'use_pe': True, 'use_lra': True, 'use_pffe': True,
        'use_ula': True, 'use_dsa': True,
        'lra_rank': 4, 'pffe_scales': [3, 5, 7],
        'ula_compression': 16, 'dsa_sparsity': 0.25,
        'filter_area_threshold': 0.8, 'filter_min_dice': 0.50,
    }
    for k, v in pe_defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    if args.dataset.lower() == 'all':
        datasets_to_test = SUPPORTED_DATASETS
        print(f"\n将测试全部 {len(datasets_to_test)} 个数据集: {datasets_to_test}\n")
    else:
        datasets_to_test = [args.dataset]

    all_results = {}
    for ds_name in datasets_to_test:
        data_path = args.data_path if args.dataset.lower() != 'all' else os.path.join(DATA_ROOT, ds_name)
        if not os.path.exists(data_path):
            print(f"警告: 数据集路径不存在，跳过 {ds_name}: {data_path}")
            continue
        try:
            summary = test_single_dataset(args, ds_name, data_path)
            if summary:
                all_results[ds_name] = summary
        except Exception as e:
            print(f"测试 {ds_name} 时出错: {e}")
            import traceback
            traceback.print_exc()

    if len(all_results) > 1:
        print("\n" + "=" * 70 + "\n全部测试完成! 汇总结果:\n" + "=" * 70)
        for ds, m in all_results.items():
            print(
                f"{ds}: Dice={m['avg_dice']:.4f}, IoU={m['avg_iou']:.4f}, HD95={m['avg_hd95']:.2f}, ASD={m['avg_asd']:.2f}")
        pd.DataFrame(all_results.values()).to_excel(os.path.join(SAVE_ROOT, 'all_datasets_summary.xlsx'), index=False)
        print(f"汇总结果已保存到: {os.path.join(SAVE_ROOT, 'all_datasets_summary.xlsx')}")


if __name__ == '__main__':
    main()