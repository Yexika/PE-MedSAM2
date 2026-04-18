# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# build_sam.py
import logging
import os

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf


def build_sam2(
        config_file,
        ckpt_path=None,
        device="cuda",
        mode="eval",
        hydra_overrides_extra=[],
        apply_postprocessing=True,
):
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor(
        config_file,
        ckpt_path=None,
        device="cuda",
        mode="eval",
        hydra_overrides_extra=[],
        apply_postprocessing=True,
):
    hydra_overrides = [
        "++model._target_=sam2_train.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _load_checkpoint(model, ckpt_path):
    """加载 checkpoint，支持跳过加载进行全量训练"""

    # ★★★ 支持跳过加载权重 ★★★
    if ckpt_path is None or ckpt_path == "" or str(ckpt_path).lower() == "none":
        print("=" * 60)
        print("★ 不加载预训练权重，从头开始全量训练 (Random Init) ★")
        print("=" * 60)
        return

    if not os.path.exists(ckpt_path):
        print(f"警告: Checkpoint 文件不存在: {ckpt_path}")
        print("将从头开始训练...")
        return

    print(f"正在加载预训练权重: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu")

    # 处理不同格式的 checkpoint
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]

    missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
    if missing_keys:
        logging.warning(f"Missing keys ({len(missing_keys)})")
    if unexpected_keys:
        logging.warning(f"Unexpected keys ({len(unexpected_keys)})")
    print("预训练权重加载完成!")