"""
func_2d/function.py - MedSAM2 2D 训练/验证函数 (改进版)

新增功能:
1. train_sam_with_scheduler: 支持学习率调度器 + 梯度裁剪
2. 保留原有 train_sam 函数兼容性
"""
#function_improved.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy import ndimage
from scipy.ndimage import distance_transform_edt

from func_2d.utils import *


# ============== Loss 函数 ==============

class BCEDiceLoss(nn.Module):
    """BCE + Dice 组合损失"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        if pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pred_sigmoid = torch.sigmoid(pred)
        smooth = 1e-5
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def get_loss_function(loss_type='bce', pos_weight=2.0, device='cuda'):
    pw = torch.ones([1]).to(device) * pos_weight
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    elif loss_type == 'dice':
        return BCEDiceLoss(bce_weight=0.0, dice_weight=1.0, pos_weight=pw)
    elif loss_type == 'bce_dice':
        return BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, pos_weight=pw)
    else:
        return nn.BCEWithLogitsLoss(pos_weight=pw)


# ============== 动态尺寸计算 ==============

def get_feat_sizes(image_size):
    return [
        (image_size // 4, image_size // 4),
        (image_size // 8, image_size // 8),
        (image_size // 16, image_size // 16)
    ]


def get_embed_size(image_size):
    return image_size // 16


# ============== 指标计算函数 ==============

def compute_dice(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    intersection = np.logical_and(pred, gt).sum()
    return (2. * intersection) / (pred.sum() + gt.sum() + 1e-6)


def compute_iou(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / (union + 1e-6)


def compute_surface_distances(pred, gt, spacing=(1.0, 1.0)):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
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
    if d_p2g is None or d_g2p is None:
        return float('inf')
    all_distances = np.concatenate([d_p2g, d_g2p])
    return np.percentile(all_distances, 95)


def compute_asd(pred, gt, spacing=(1.0, 1.0)):
    d_p2g, d_g2p = compute_surface_distances(pred, gt, spacing)
    if d_p2g is None or d_g2p is None:
        return float('inf')
    return (np.mean(d_p2g) + np.mean(d_g2p)) / 2


# ============== 训练函数 (带调度器和梯度裁剪) ==============

def train_sam_with_scheduler(args, net: nn.Module, optimizer, train_loader, epoch, writer,
                              scheduler=None, max_grad_norm=1.0, global_step=0, lossfunc=None):
    """
    改进版训练函数
    
    新增参数:
        scheduler: 学习率调度器 (每步更新)
        max_grad_norm: 梯度裁剪阈值
        global_step: 全局步数 (用于调度器)
    
    返回:
        epoch_loss: 平均损失
        global_step: 更新后的全局步数
    """
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    net.train()
    optimizer.zero_grad()

    GPUdevice = torch.device('cuda', args.gpu_device)
    mask_type = torch.float32

    if lossfunc is None:
        loss_type = getattr(args, 'loss_type', 'bce')
        lossfunc = get_loss_function(loss_type, pos_weight=2.0, device=GPUdevice)

    epoch_loss = 0
    memory_bank_list = []

    feat_sizes = get_feat_sizes(args.image_size)
    embed_size = get_embed_size(args.image_size)

    if epoch == 0:
        print(f"[Train] image_size={args.image_size}, feat_sizes={feat_sizes}")
        print(f"[Train] Loss: {lossfunc.__class__.__name__}")
        print(f"[Train] 梯度裁剪阈值: {max_grad_norm}")
        if scheduler:
            print(f"[Train] 使用学习率调度器: 每步更新")

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for ind, pack in enumerate(train_loader):
            to_cat_memory = []
            to_cat_memory_pos = []
            to_cat_image_embed = []

            imgs = pack['image'].to(dtype=mask_type, device=GPUdevice)
            masks = pack['mask'].to(dtype=mask_type, device=GPUdevice)
            name = pack['image_meta_dict']['filename_or_obj']

            if 'pt' in pack:
                pt_temp = pack['pt'].to(device=GPUdevice)
                pt = pt_temp.unsqueeze(1)
                point_labels_temp = pack['p_label'].to(device=GPUdevice)
                point_labels = point_labels_temp.unsqueeze(1)
                coords_torch = torch.as_tensor(pt, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            else:
                coords_torch = None
                labels_torch = None

            backbone_out = net.forward_image(imgs)
            _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
            B = vision_feats[-1].size(1)

            if len(memory_bank_list) == 0:
                vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(
                    torch.zeros(1, B, net.hidden_dim)).to(device=GPUdevice)
                vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(
                    torch.zeros(1, B, net.hidden_dim)).to(device=GPUdevice)
            else:
                for element in memory_bank_list:
                    to_cat_memory.append(element[0].to(GPUdevice, non_blocking=True).flatten(2).permute(2, 0, 1))
                    to_cat_memory_pos.append(element[1].to(GPUdevice, non_blocking=True).flatten(2).permute(2, 0, 1))
                    to_cat_image_embed.append(element[3].to(GPUdevice, non_blocking=True))

                memory_stack_ori = torch.stack(to_cat_memory, dim=0)
                memory_pos_stack_ori = torch.stack(to_cat_memory_pos, dim=0)
                image_embed_stack_ori = torch.stack(to_cat_image_embed, dim=0)

                vision_feats_temp = vision_feats[-1].permute(1, 0, 2).reshape(B, -1, embed_size, embed_size)
                vision_feats_temp = vision_feats_temp.reshape(B, -1)

                image_embed_stack_ori = F.normalize(image_embed_stack_ori, p=2, dim=1)
                vision_feats_temp = F.normalize(vision_feats_temp, p=2, dim=1)
                similarity_scores = torch.mm(image_embed_stack_ori, vision_feats_temp.t()).t()
                similarity_scores = F.softmax(similarity_scores, dim=1)
                sampled_indices = torch.multinomial(similarity_scores, num_samples=B, replacement=True).squeeze(1)

                memory_stack_ori_new = memory_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3)
                memory = memory_stack_ori_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))
                memory_pos_stack_new = memory_pos_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3)
                memory_pos = memory_pos_stack_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))

                vision_feats[-1] = net.memory_attention(
                    curr=[vision_feats[-1]], curr_pos=[vision_pos_embeds[-1]],
                    memory=memory, memory_pos=memory_pos, num_obj_ptr_tokens=0
                )

            feats = [feat.permute(1, 2, 0).view(B, -1, *feat_size)
                     for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
            image_embed = feats[-1]
            high_res_feats = feats[:-1]

            with torch.no_grad():
                if (ind % 5) == 0:
                    points = (coords_torch, labels_torch)
                    flag = True
                else:
                    points = None
                    flag = False

                se, de = net.sam_prompt_encoder(points=points, boxes=None, masks=None, batch_size=B)

            low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = net.sam_mask_decoder(
                image_embeddings=image_embed, image_pe=net.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se, dense_prompt_embeddings=de,
                multimask_output=False, repeat_image=False, high_res_features=high_res_feats
            )

            pred = F.interpolate(low_res_multimasks, size=(args.out_size, args.out_size))
            high_res_multimasks = F.interpolate(low_res_multimasks, size=(args.image_size, args.image_size),
                                                mode="bilinear", align_corners=False)

            maskmem_features, maskmem_pos_enc = net._encode_new_memory(
                current_vision_feats=vision_feats, feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_multimasks, is_mask_from_pts=flag)

            maskmem_features = maskmem_features.to(torch.bfloat16).to(device=GPUdevice, non_blocking=True)
            maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16).to(device=GPUdevice, non_blocking=True)

            # Memory bank 更新
            if len(memory_bank_list) < args.memory_bank_size:
                for batch in range(maskmem_features.size(0)):
                    memory_bank_list.append([
                        maskmem_features[batch].unsqueeze(0).detach(),
                        maskmem_pos_enc[batch].unsqueeze(0).detach(),
                        iou_predictions[batch, 0],
                        image_embed[batch].reshape(-1).detach()
                    ])
            else:
                for batch in range(maskmem_features.size(0)):
                    memory_bank_maskmem_features_flatten = torch.stack([e[0].reshape(-1) for e in memory_bank_list])
                    memory_bank_maskmem_features_norm = F.normalize(memory_bank_maskmem_features_flatten, p=2, dim=1)
                    current_similarity_matrix = torch.mm(memory_bank_maskmem_features_norm, memory_bank_maskmem_features_norm.t())
                    current_similarity_matrix_no_diag = current_similarity_matrix.clone()
                    diag_indices = torch.arange(current_similarity_matrix_no_diag.size(0))
                    current_similarity_matrix_no_diag[diag_indices, diag_indices] = float('-inf')

                    single_key_norm = F.normalize(maskmem_features[batch].reshape(-1), p=2, dim=0).unsqueeze(1)
                    similarity_scores = torch.mm(memory_bank_maskmem_features_norm, single_key_norm).squeeze()
                    min_similarity_index = torch.argmin(similarity_scores)
                    max_similarity_index = torch.argmax(current_similarity_matrix_no_diag[min_similarity_index])

                    if similarity_scores[min_similarity_index] < current_similarity_matrix_no_diag[min_similarity_index][max_similarity_index]:
                        if iou_predictions[batch, 0] > memory_bank_list[max_similarity_index][2] - 0.1:
                            memory_bank_list.pop(max_similarity_index)
                            memory_bank_list.append([
                                maskmem_features[batch].unsqueeze(0).detach(),
                                maskmem_pos_enc[batch].unsqueeze(0).detach(),
                                iou_predictions[batch, 0],
                                image_embed[batch].reshape(-1).detach()
                            ])

            loss = lossfunc(pred, masks)
            epoch_loss += loss.item()

            # 反向传播
            loss.backward()
            
            # ★★★ 梯度裁剪 ★★★
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # ★★★ 学习率调度器更新 (每步) ★★★
            if scheduler is not None:
                scheduler.step()
            
            global_step += 1
            
            # 更新进度条
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(**{
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}'
            })
            pbar.update()
            
            # 记录到 TensorBoard
            if global_step % 10 == 0:
                writer.add_scalar('train/loss_step', loss.item(), global_step)
                writer.add_scalar('train/lr_step', current_lr, global_step)

    return epoch_loss / len(train_loader), global_step


# ============== 原始训练函数 (保持兼容) ==============

def train_sam(args, net: nn.Module, optimizer, train_loader, epoch, writer, lossfunc=None):
    """原始训练函数 - 保持向后兼容"""
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    net.train()
    optimizer.zero_grad()

    GPUdevice = torch.device('cuda', args.gpu_device)
    mask_type = torch.float32

    if lossfunc is None:
        loss_type = getattr(args, 'loss_type', 'bce')
        lossfunc = get_loss_function(loss_type, pos_weight=2.0, device=GPUdevice)

    epoch_loss = 0
    memory_bank_list = []

    feat_sizes = get_feat_sizes(args.image_size)
    embed_size = get_embed_size(args.image_size)

    if epoch == 0:
        print(f"[Train] image_size={args.image_size}, feat_sizes={feat_sizes}, embed_size={embed_size}")
        print(f"[Train] Loss: {lossfunc.__class__.__name__}")

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for ind, pack in enumerate(train_loader):
            to_cat_memory = []
            to_cat_memory_pos = []
            to_cat_image_embed = []

            imgs = pack['image'].to(dtype=mask_type, device=GPUdevice)
            masks = pack['mask'].to(dtype=mask_type, device=GPUdevice)
            name = pack['image_meta_dict']['filename_or_obj']

            if 'pt' in pack:
                pt_temp = pack['pt'].to(device=GPUdevice)
                pt = pt_temp.unsqueeze(1)
                point_labels_temp = pack['p_label'].to(device=GPUdevice)
                point_labels = point_labels_temp.unsqueeze(1)
                coords_torch = torch.as_tensor(pt, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            else:
                coords_torch = None
                labels_torch = None

            backbone_out = net.forward_image(imgs)
            _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
            B = vision_feats[-1].size(1)

            if len(memory_bank_list) == 0:
                vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(
                    torch.zeros(1, B, net.hidden_dim)).to(device=GPUdevice)
                vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(
                    torch.zeros(1, B, net.hidden_dim)).to(device=GPUdevice)
            else:
                for element in memory_bank_list:
                    to_cat_memory.append(element[0].to(GPUdevice, non_blocking=True).flatten(2).permute(2, 0, 1))
                    to_cat_memory_pos.append(element[1].to(GPUdevice, non_blocking=True).flatten(2).permute(2, 0, 1))
                    to_cat_image_embed.append(element[3].to(GPUdevice, non_blocking=True))

                memory_stack_ori = torch.stack(to_cat_memory, dim=0)
                memory_pos_stack_ori = torch.stack(to_cat_memory_pos, dim=0)
                image_embed_stack_ori = torch.stack(to_cat_image_embed, dim=0)

                vision_feats_temp = vision_feats[-1].permute(1, 0, 2).reshape(B, -1, embed_size, embed_size)
                vision_feats_temp = vision_feats_temp.reshape(B, -1)

                image_embed_stack_ori = F.normalize(image_embed_stack_ori, p=2, dim=1)
                vision_feats_temp = F.normalize(vision_feats_temp, p=2, dim=1)
                similarity_scores = torch.mm(image_embed_stack_ori, vision_feats_temp.t()).t()
                similarity_scores = F.softmax(similarity_scores, dim=1)
                sampled_indices = torch.multinomial(similarity_scores, num_samples=B, replacement=True).squeeze(1)

                memory_stack_ori_new = memory_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3)
                memory = memory_stack_ori_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))
                memory_pos_stack_new = memory_pos_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3)
                memory_pos = memory_pos_stack_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))

                vision_feats[-1] = net.memory_attention(
                    curr=[vision_feats[-1]], curr_pos=[vision_pos_embeds[-1]],
                    memory=memory, memory_pos=memory_pos, num_obj_ptr_tokens=0
                )

            feats = [feat.permute(1, 2, 0).view(B, -1, *feat_size)
                     for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
            image_embed = feats[-1]
            high_res_feats = feats[:-1]

            with torch.no_grad():
                if (ind % 5) == 0:
                    points = (coords_torch, labels_torch)
                    flag = True
                else:
                    points = None
                    flag = False

                se, de = net.sam_prompt_encoder(points=points, boxes=None, masks=None, batch_size=B)

            low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = net.sam_mask_decoder(
                image_embeddings=image_embed, image_pe=net.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se, dense_prompt_embeddings=de,
                multimask_output=False, repeat_image=False, high_res_features=high_res_feats
            )

            pred = F.interpolate(low_res_multimasks, size=(args.out_size, args.out_size))
            high_res_multimasks = F.interpolate(low_res_multimasks, size=(args.image_size, args.image_size),
                                                mode="bilinear", align_corners=False)

            maskmem_features, maskmem_pos_enc = net._encode_new_memory(
                current_vision_feats=vision_feats, feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_multimasks, is_mask_from_pts=flag)

            maskmem_features = maskmem_features.to(torch.bfloat16).to(device=GPUdevice, non_blocking=True)
            maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16).to(device=GPUdevice, non_blocking=True)

            if len(memory_bank_list) < args.memory_bank_size:
                for batch in range(maskmem_features.size(0)):
                    memory_bank_list.append([
                        maskmem_features[batch].unsqueeze(0).detach(),
                        maskmem_pos_enc[batch].unsqueeze(0).detach(),
                        iou_predictions[batch, 0],
                        image_embed[batch].reshape(-1).detach()
                    ])
            else:
                for batch in range(maskmem_features.size(0)):
                    memory_bank_maskmem_features_flatten = torch.stack([e[0].reshape(-1) for e in memory_bank_list])
                    memory_bank_maskmem_features_norm = F.normalize(memory_bank_maskmem_features_flatten, p=2, dim=1)
                    current_similarity_matrix = torch.mm(memory_bank_maskmem_features_norm, memory_bank_maskmem_features_norm.t())
                    current_similarity_matrix_no_diag = current_similarity_matrix.clone()
                    diag_indices = torch.arange(current_similarity_matrix_no_diag.size(0))
                    current_similarity_matrix_no_diag[diag_indices, diag_indices] = float('-inf')

                    single_key_norm = F.normalize(maskmem_features[batch].reshape(-1), p=2, dim=0).unsqueeze(1)
                    similarity_scores = torch.mm(memory_bank_maskmem_features_norm, single_key_norm).squeeze()
                    min_similarity_index = torch.argmin(similarity_scores)
                    max_similarity_index = torch.argmax(current_similarity_matrix_no_diag[min_similarity_index])

                    if similarity_scores[min_similarity_index] < current_similarity_matrix_no_diag[min_similarity_index][max_similarity_index]:
                        if iou_predictions[batch, 0] > memory_bank_list[max_similarity_index][2] - 0.1:
                            memory_bank_list.pop(max_similarity_index)
                            memory_bank_list.append([
                                maskmem_features[batch].unsqueeze(0).detach(),
                                maskmem_pos_enc[batch].unsqueeze(0).detach(),
                                iou_predictions[batch, 0],
                                image_embed[batch].reshape(-1).detach()
                            ])

            loss = lossfunc(pred, masks)
            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.update()

    return epoch_loss / len(train_loader)


# ============== 验证函数 ==============

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True, lossfunc=None):
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    net.eval()
    n_val = len(val_loader)
    GPUdevice = torch.device('cuda', args.gpu_device)

    if lossfunc is None:
        loss_type = getattr(args, 'loss_type', 'bce')
        lossfunc = get_loss_function(loss_type, pos_weight=2.0, device=GPUdevice)

    memory_bank_list = []

    feat_sizes = get_feat_sizes(args.image_size)
    embed_size = get_embed_size(args.image_size)

    total_loss = 0
    per_sample_results = []
    total_intersection = 0
    total_union = 0
    total_pred_sum = 0
    total_gt_sum = 0
    all_pred_to_gt = []
    all_gt_to_pred = []

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            to_cat_memory = []
            to_cat_memory_pos = []
            to_cat_image_embed = []

            name = pack['image_meta_dict']['filename_or_obj']
            imgs = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masks = pack['mask'].to(dtype=torch.float32, device=GPUdevice)

            if 'pt' in pack:
                pt_temp = pack['pt'].to(device=GPUdevice)
                pt = pt_temp.unsqueeze(1)
                point_labels_temp = pack['p_label'].to(device=GPUdevice)
                point_labels = point_labels_temp.unsqueeze(1)
                coords_torch = torch.as_tensor(pt, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            else:
                coords_torch = None
                labels_torch = None

            with torch.no_grad():
                backbone_out = net.forward_image(imgs)
                _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
                B = vision_feats[-1].size(1)

                if len(memory_bank_list) == 0:
                    vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(
                        torch.zeros(1, B, net.hidden_dim)).to(device=GPUdevice)
                    vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(
                        torch.zeros(1, B, net.hidden_dim)).to(device=GPUdevice)
                else:
                    for element in memory_bank_list:
                        to_cat_memory.append(element[0].to(GPUdevice, non_blocking=True).flatten(2).permute(2, 0, 1))
                        to_cat_memory_pos.append(element[1].to(GPUdevice, non_blocking=True).flatten(2).permute(2, 0, 1))
                        to_cat_image_embed.append(element[3].to(GPUdevice, non_blocking=True))

                    memory_stack_ori = torch.stack(to_cat_memory, dim=0)
                    memory_pos_stack_ori = torch.stack(to_cat_memory_pos, dim=0)
                    image_embed_stack_ori = torch.stack(to_cat_image_embed, dim=0)

                    vision_feats_temp = vision_feats[-1].permute(1, 0, 2).view(B, -1, embed_size, embed_size)
                    vision_feats_temp = vision_feats_temp.reshape(B, -1)

                    image_embed_stack_ori = F.normalize(image_embed_stack_ori, p=2, dim=1)
                    vision_feats_temp = F.normalize(vision_feats_temp, p=2, dim=1)
                    similarity_scores = torch.mm(image_embed_stack_ori, vision_feats_temp.t()).t()
                    similarity_scores = F.softmax(similarity_scores, dim=1)
                    sampled_indices = torch.multinomial(similarity_scores, num_samples=B, replacement=True).squeeze(1)

                    memory_stack_ori_new = memory_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3)
                    memory = memory_stack_ori_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))
                    memory_pos_stack_new = memory_pos_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3)
                    memory_pos = memory_pos_stack_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))

                    vision_feats[-1] = net.memory_attention(
                        curr=[vision_feats[-1]], curr_pos=[vision_pos_embeds[-1]],
                        memory=memory, memory_pos=memory_pos, num_obj_ptr_tokens=0
                    )

                feats = [feat.permute(1, 2, 0).view(B, -1, *feat_size)
                         for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
                image_embed = feats[-1]
                high_res_feats = feats[:-1]

                if (ind % 5) == 0:
                    flag = True
                    points = (coords_torch, labels_torch)
                else:
                    flag = False
                    points = None

                se, de = net.sam_prompt_encoder(points=points, boxes=None, masks=None, batch_size=B)

                low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = net.sam_mask_decoder(
                    image_embeddings=image_embed, image_pe=net.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se, dense_prompt_embeddings=de,
                    multimask_output=False, repeat_image=False, high_res_features=high_res_feats
                )

                pred = F.interpolate(low_res_multimasks, size=(args.out_size, args.out_size))
                high_res_multimasks = F.interpolate(low_res_multimasks, size=(args.image_size, args.image_size),
                                                    mode="bilinear", align_corners=False)

                maskmem_features, maskmem_pos_enc = net._encode_new_memory(
                    current_vision_feats=vision_feats, feat_sizes=feat_sizes,
                    pred_masks_high_res=high_res_multimasks, is_mask_from_pts=flag)

                maskmem_features = maskmem_features.to(torch.bfloat16).to(device=GPUdevice, non_blocking=True)
                maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16).to(device=GPUdevice, non_blocking=True)

                if len(memory_bank_list) < 16:
                    for batch in range(maskmem_features.size(0)):
                        memory_bank_list.append([
                            maskmem_features[batch].unsqueeze(0),
                            maskmem_pos_enc[batch].unsqueeze(0),
                            iou_predictions[batch, 0],
                            image_embed[batch].reshape(-1).detach()
                        ])
                else:
                    for batch in range(maskmem_features.size(0)):
                        memory_bank_maskmem_features_flatten = torch.stack([e[0].reshape(-1) for e in memory_bank_list])
                        memory_bank_maskmem_features_norm = F.normalize(memory_bank_maskmem_features_flatten, p=2, dim=1)
                        current_similarity_matrix = torch.mm(memory_bank_maskmem_features_norm, memory_bank_maskmem_features_norm.t())
                        current_similarity_matrix_no_diag = current_similarity_matrix.clone()
                        diag_indices = torch.arange(current_similarity_matrix_no_diag.size(0))
                        current_similarity_matrix_no_diag[diag_indices, diag_indices] = float('-inf')

                        single_key_norm = F.normalize(maskmem_features[batch].reshape(-1), p=2, dim=0).unsqueeze(1)
                        similarity_scores = torch.mm(memory_bank_maskmem_features_norm, single_key_norm).squeeze()
                        min_similarity_index = torch.argmin(similarity_scores)
                        max_similarity_index = torch.argmax(current_similarity_matrix_no_diag[min_similarity_index])

                        if similarity_scores[min_similarity_index] < current_similarity_matrix_no_diag[min_similarity_index][max_similarity_index]:
                            if iou_predictions[batch, 0] > memory_bank_list[max_similarity_index][2] - 0.1:
                                memory_bank_list.pop(max_similarity_index)
                                memory_bank_list.append([
                                    maskmem_features[batch].unsqueeze(0),
                                    maskmem_pos_enc[batch].unsqueeze(0),
                                    iou_predictions[batch, 0],
                                    image_embed[batch].reshape(-1).detach()
                                ])

                total_loss += lossfunc(pred, masks)
                pred_binary = (pred > 0).float()

                for b in range(B):
                    sample_name = name[b] if isinstance(name, (list, tuple)) else name
                    pred_np = pred_binary[b, 0].cpu().numpy()
                    gt_np = masks[b, 0].cpu().numpy()

                    dice = compute_dice(pred_np, gt_np)
                    iou_val = compute_iou(pred_np, gt_np)
                    hd95 = compute_hd95(pred_np, gt_np)
                    asd = compute_asd(pred_np, gt_np)

                    per_sample_results.append({
                        'name': sample_name, 'dice': dice, 'iou': iou_val,
                        'hd95': hd95 if not np.isinf(hd95) else -1,
                        'asd': asd if not np.isinf(asd) else -1
                    })

                    pred_flat = pred_np.flatten().astype(bool)
                    gt_flat = gt_np.flatten().astype(bool)
                    total_intersection += np.sum(pred_flat & gt_flat)
                    total_union += np.sum(pred_flat | gt_flat)
                    total_pred_sum += np.sum(pred_flat)
                    total_gt_sum += np.sum(gt_flat)

                    d_p2g, d_g2p = compute_surface_distances(pred_np, gt_np)
                    if d_p2g is not None and d_g2p is not None:
                        all_pred_to_gt.extend(d_p2g.tolist())
                        all_gt_to_pred.extend(d_g2p.tolist())

                if ind % args.vis == 0:
                    namecat = 'Test'
                    for na in name:
                        namecat = namecat + na + '+'
                    vis_image(imgs, pred_binary, masks,
                              os.path.join(args.path_helper['sample_path'], namecat + 'epoch+' + str(epoch) + '.jpg'),
                              reverse=False, points=None)

            pbar.update()

    smooth = 1e-6
    global_dice = (2 * total_intersection + smooth) / (total_pred_sum + total_gt_sum + smooth)
    global_iou = (total_intersection + smooth) / (total_union + smooth)

    if len(all_pred_to_gt) > 0 and len(all_gt_to_pred) > 0:
        all_distances = np.array(all_pred_to_gt + all_gt_to_pred)
        global_hd95 = np.percentile(all_distances, 95)
        global_asd = (np.mean(all_pred_to_gt) + np.mean(all_gt_to_pred)) / 2
    else:
        global_hd95 = float('inf')
        global_asd = float('inf')

    valid_hd95 = [r['hd95'] for r in per_sample_results if r['hd95'] >= 0]
    valid_asd = [r['asd'] for r in per_sample_results if r['asd'] >= 0]

    avg_dice = np.mean([r['dice'] for r in per_sample_results])
    avg_iou = np.mean([r['iou'] for r in per_sample_results])
    avg_hd95 = np.mean(valid_hd95) if valid_hd95 else float('inf')
    avg_asd = np.mean(valid_asd) if valid_asd else float('inf')

    print(f"\n{'=' * 60}")
    print(f"Epoch {epoch} 验证结果 (image_size={args.image_size})")
    print(f"{'=' * 60}")
    print(f"[全局指标] Dice: {global_dice:.4f} | IoU: {global_iou:.4f} | HD95: {global_hd95:.2f} | ASD: {global_asd:.2f}")
    print(f"[平均指标] Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f} | HD95: {avg_hd95:.2f} | ASD: {avg_asd:.2f}")
    print(f"{'=' * 60}\n")

    return total_loss / n_val, (avg_iou, avg_dice, global_dice, global_iou, avg_hd95, avg_asd, global_hd95, global_asd)
