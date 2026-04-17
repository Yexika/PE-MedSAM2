"""
Fixed Dataset Class for MedSAM2 2D
修复版本 v3 - 简化处理逻辑

核心改进:
1. Image 和 Mask 使用相同的双三次插值 (BICUBIC)
2. 单mask数据集只需一次二值化
3. REFUGE多专家数据集保持投票逻辑

func_2d/dataset_modified.py
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from func_2d.utils import random_click  # 直接用官方的


# ------------------------------
# Utilities
# ------------------------------

def _ensure_dir(path: str):
    if not os.path.isdir(path):
        raise ValueError(f"Directory not found: {path}")


def _list_images(dir_path: str):
    exts = ('.png', '.jpg', '.jpeg', '.bmp')
    return sorted([f for f in os.listdir(dir_path) if f.lower().endswith(exts)])


def _split_file_path(root: str, seed: int):
    return os.path.join(root, f"split_8_1_1_seed{seed}.json")


def _load_or_create_split(root: str, ids: list, seed: int):
    split_path = _split_file_path(root, seed)
    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # create new split
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(ids))
    n = len(ids)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    split = {
        "train": [ids[i] for i in perm[:n_train]],
        "val":   [ids[i] for i in perm[n_train:n_train + n_val]],
        "test":  [ids[i] for i in perm[n_train + n_val:]],
    }
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    return split


# ------------------------------
# Generic PNG Dataset (Fixed v2)
# ------------------------------

class MultiDataset(Dataset):
    """
    单mask数据集 - 简化版

    关键点:
    1. Image 和 Mask 使用相同的双三次插值 (BICUBIC)
    2. Mask 只需一次二值化（没有多专家投票）
    """

    def __init__(self, args, data_path, mode='Training', prompt='click', seed=1234, enable_aug=True):
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = int(args.image_size)
        self.mask_size = int(args.out_size)

        img_dir = os.path.join(data_path, 'images')
        msk_dir = os.path.join(data_path, 'masks')
        _ensure_dir(img_dir)
        _ensure_dir(msk_dir)

        img_files = _list_images(img_dir)
        if len(img_files) == 0:
            raise ValueError(f"No images found in {img_dir}")
        ids = [os.path.splitext(f)[0] for f in img_files]

        split = _load_or_create_split(data_path, ids, seed)
        if mode == 'Training':
            self.ids = split['train']
        elif mode == 'Validation':
            self.ids = split['val']
        elif mode == 'Test':
            self.ids = split['test']
        else:
            raise ValueError("mode must be 'Training' | 'Validation' | 'Test'")

        self.image_dir = img_dir
        self.mask_dir = msk_dir
        self.enable_aug = enable_aug and (mode == 'Training')

        print(f"[{mode}] {len(self.ids)} samples | img_size={self.img_size}, mask_size={self.mask_size}")

    def __len__(self):
        return len(self.ids)

    def _resolve_mask_path(self, base: str):
        """Try to find mask file with various naming conventions."""
        for ext in ('.png', '.jpg', '.jpeg', '.bmp'):
            candidates = [
                os.path.join(self.mask_dir, base + ext),
                os.path.join(self.mask_dir, base + '_mask' + ext),
                os.path.join(self.mask_dir, base + '_gt' + ext),
                os.path.join(self.mask_dir, base + '_segmentation' + ext),
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    return cand
        return os.path.join(self.mask_dir, base + '.png')

    def _find_image_path(self, sid: str):
        for ext in ('.png', '.jpg', '.jpeg', '.bmp'):
            p = os.path.join(self.image_dir, sid + ext)
            if os.path.exists(p):
                return p
        return None

    def __getitem__(self, index):
        sid = self.ids[index]

        img_path = self._find_image_path(sid)
        if img_path is None:
            raise FileNotFoundError(f"Image not found for id={sid} in {self.image_dir}")

        msk_path = self._resolve_mask_path(sid)
        if not os.path.exists(msk_path):
            raise FileNotFoundError(f"Mask not found for id={sid} -> {msk_path}")

        # 加载图像和 mask
        img = Image.open(img_path).convert('RGB')
        msk = Image.open(msk_path).convert('L')

        # 数据增强（在resize之前）
        if self.enable_aug:
            if np.random.rand() < 0.5:
                img = TF.hflip(img)
                msk = TF.hflip(msk)
            if np.random.rand() < 0.5:
                img = TF.vflip(img)
                msk = TF.vflip(msk)

        # ★★★ 双三次插值 - Image 和 Mask 使用相同方式 ★★★
        img = TF.resize(img, [self.img_size, self.img_size], interpolation=Image.BICUBIC)
        msk = TF.resize(msk, [self.img_size, self.img_size], interpolation=Image.BICUBIC)

        # 转 tensor
        img_t = TF.to_tensor(img)  # [3, H, W]
        msk_t = TF.to_tensor(msk)  # [1, H, W]
        msk_t = (msk_t >= 0.5).float()  # 一次二值化即可

        # 输出尺寸的 mask
        if self.img_size != self.mask_size:
            msk_out = F.interpolate(
                msk_t.unsqueeze(0),
                size=(self.mask_size, self.mask_size),
                mode='bicubic',
                align_corners=False
            ).squeeze(0)
            msk_out = (msk_out >= 0.5).float()
        else:
            msk_out = msk_t

        # 点击提示
        if self.prompt == 'click':
            p_label, pt = random_click(msk_t.squeeze(0).numpy(), point_label=1)
        else:
            p_label, pt = 1, np.array([self.img_size // 2, self.img_size // 2])

        image_meta_dict = {'filename_or_obj': sid}

        return {
            'image': img_t,              # [3, img_size, img_size]
            'mask': msk_out,             # [1, mask_size, mask_size]
            'mask_ori': msk_t,           # [1, img_size, img_size]
            'p_label': p_label,          # int
            'pt': pt,                    # np.ndarray [y, x]
            'image_meta_dict': image_meta_dict,
        }


# ------------------------------
# Factory
# ------------------------------

def get_dataset(args, data_path, mode='Training', seed=1234):
    """
    工厂函数：
    - REFUGE 数据集用官方的 dataset.py
    - 其他数据集用 MultiDataset
    """
    if getattr(args, 'dataset', '').upper() == 'REFUGE':
        from func_2d.dataset import REFUGE
        return REFUGE(args, data_path, mode=mode, prompt='click')
    else:
        return MultiDataset(args, data_path, mode=mode, seed=seed)