#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
func_2d/filter_utils.py - 异常预测过滤工具
"""
import numpy as np


def filter_abnormal_prediction(pred_mask, gt_mask,
                               area_threshold=0.9,
                               min_dice_threshold=0.05):
    """
    过滤异常预测（预测区域过大的情况）

    Args:
        pred_mask: 预测mask (H, W) 二值numpy数组
        gt_mask: 真实mask (H, W) 二值numpy数组
        area_threshold: 预测区域占比阈值（默认0.9，即90%）
        min_dice_threshold: 最小dice阈值（默认0.05）

    Returns:
        filtered_pred: 过滤后的mask
        is_abnormal: 是否为异常预测
        reason: 异常原因（如果is_abnormal=True）
    """
    total_pixels = pred_mask.shape[0] * pred_mask.shape[1]
    pred_pixels = pred_mask.sum()
    gt_pixels = gt_mask.sum()

    pred_ratio = pred_pixels / total_pixels
    gt_ratio = gt_pixels / total_pixels

    # 情况1: 预测区域过大（>90%），且GT区域较小（<50%）
    if pred_ratio > area_threshold and gt_ratio < 0.5:
        return np.zeros_like(pred_mask), True, f"pred_area={pred_ratio:.2%}, gt_area={gt_ratio:.2%}"

    # 情况2: 快速计算Dice，如果极低且预测区域很大，可能是失败案例
    if pred_pixels > 0 and gt_pixels > 0:
        intersection = (pred_mask * gt_mask).sum()
        dice = 2 * intersection / (pred_pixels + gt_pixels)

        if dice < min_dice_threshold and pred_ratio > 0.8:
            return np.zeros_like(pred_mask), True, f"low_dice={dice:.4f}, pred_area={pred_ratio:.2%}"

    return pred_mask, False, ""


class AbnormalStats:
    """统计异常预测的工具类"""

    def __init__(self):
        self.total = 0
        self.abnormal = 0
        self.abnormal_details = []

    def update(self, sample_name, is_abnormal, reason=""):
        self.total += 1
        if is_abnormal:
            self.abnormal += 1
            self.abnormal_details.append({
                'name': sample_name,
                'reason': reason
            })

    def get_ratio(self):
        return self.abnormal / self.total if self.total > 0 else 0

    def summary(self):
        ratio = self.get_ratio()
        return f"异常预测: {self.abnormal}/{self.total} ({ratio * 100:.2f}%)"

    def print_details(self):
        if self.abnormal > 0:
            print(f"\n{'=' * 60}")
            print(f"异常预测详情 ({self.abnormal} 个样本):")
            print(f"{'=' * 60}")
            for item in self.abnormal_details[:10]:  # 只打印前10个
                print(f"  - {item['name']}: {item['reason']}")
            if len(self.abnormal_details) > 10:
                print(f"  ... 还有 {len(self.abnormal_details) - 10} 个异常样本")
            print(f"{'=' * 60}\n")