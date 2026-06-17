# -*- coding: utf-8 -*-
"""
dataset.py —— 数据加载层 (loaddata)。

职责:
  - 定义训练/验证的图像预处理(transform), 训练带数据增强
  - 用 ImageFolder 读 data/ 下的 4 个类别子文件夹
  - 把文件夹名(可能含空格)映射到标准 CLASS_NAMES 的 index 顺序
  - 按 VAL_RATIO 划分 train/val, 返回两个 DataLoader
"""

import copy

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src import settings
from src.utils import normalize_label


def get_transforms(train: bool):
    """构建预处理。train=True 时加数据增强(小数据缓解过拟合的关键)。"""
    size = settings.IMG_SIZE
    if train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(settings.MEAN, settings.STD),
        ])
    # 验证/测试: 不增强, 只 resize + 归一化(必须与训练验证集一致)
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(settings.MEAN, settings.STD),
    ])


def build_dataloaders():
    """读取 data/, 划分 train/val, 返回 (train_loader, val_loader, info)。"""
    # 先不带 transform 读一遍, 拿到 class_to_idx 以建立标签映射
    full = datasets.ImageFolder(settings.DATA_DIR)

    # ImageFolder 内部 index -> 标准下划线标签 -> CLASS_NAMES 统一 index
    folder_to_std = {idx: normalize_label(name)
                     for name, idx in full.class_to_idx.items()}
    remap = {idx: settings.CLASS_NAMES.index(std)
             for idx, std in folder_to_std.items()}
    # 用 target_transform 把标签直接重映射到 CLASS_NAMES 顺序
    full.target_transform = lambda y: remap[y]

    # 划分 train/val
    n_val = int(len(full) * settings.VAL_RATIO)
    n_train = len(full) - n_val
    g = torch.Generator().manual_seed(settings.SEED)
    train_set, val_set = random_split(full, [n_train, n_val], generator=g)

    # 给两个子集套各自的 transform(浅拷贝底层 dataset 再分别赋 transform)
    train_set.dataset = copy.copy(full)
    train_set.dataset.transform = get_transforms(train=True)
    val_set.dataset = copy.copy(full)
    val_set.dataset.transform = get_transforms(train=False)

    train_loader = DataLoader(train_set, batch_size=settings.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=settings.BATCH_SIZE, shuffle=False)

    info = {
        "total": len(full), "n_train": n_train, "n_val": n_val,
        "folder_to_label": {name: normalize_label(name) for name in full.classes},
    }
    return train_loader, val_loader, info
