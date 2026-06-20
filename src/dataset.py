# -*- coding: utf-8 -*-
"""
dataset.py —— 数据加载层 (loaddata)。

数据布局: images/labels 同名配对
  data/images/0001.jpg  <->  data/labels/0001.txt
  txt 内容为类别字符串(book_close / book_open / laptop_close / laptop_open)。

职责:
  - 定义训练/验证的图像预处理(transform), 训练带数据增强
  - 扫描 images/ 与 labels/, 按 stem 配对, 把标签字符串映射到 CLASS_NAMES 的 index
  - 配对异常(有图无标签/有标签无图/标签非法/损坏图)收集进 skipped, 跳过而非崩溃
  - 按 VAL_RATIO 划分 train/val, 返回两个 DataLoader
"""

import os

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

from src import settings
from src.utils import normalize_label

# 支持的图片扩展名(与 test.py 保持一致)
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


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


def scan_pairs(images_dir, labels_dir):
    """扫描 images/labels, 按 stem 配对。
    返回 (samples, skipped):
      samples: [(img_path, label_idx), ...] 合法配对
      skipped: [(name, 原因), ...] 异常项, 供 EDA / 启动告警使用
    """
    samples, skipped = [], []
    if not os.path.isdir(images_dir):
        return samples, [(images_dir, "images 目录不存在")]
    if not os.path.isdir(labels_dir):
        return samples, [(labels_dir, "labels 目录不存在")]

    imgs = sorted(f for f in os.listdir(images_dir)
                  if f.lower().endswith(IMG_EXT))
    label_stems = {os.path.splitext(f)[0] for f in os.listdir(labels_dir)
                   if f.lower().endswith(".txt")}

    for fn in imgs:
        stem = os.path.splitext(fn)[0]
        img_path = os.path.join(images_dir, fn)
        lbl_path = os.path.join(labels_dir, stem + ".txt")
        if stem not in label_stems:
            skipped.append((fn, "缺少同名标签 txt"))
            continue
        with open(lbl_path) as f:
            label = normalize_label(f.read().strip())
        if label not in settings.CLASS_NAMES:
            skipped.append((fn, f"标签非法: '{label}'"))
            continue
        samples.append((img_path, settings.CLASS_NAMES.index(label)))

    # 有标签但无对应图片的, 也记一笔
    img_stems = {os.path.splitext(f)[0] for f in imgs}
    for stem in sorted(label_stems - img_stems):
        skipped.append((stem + ".txt", "缺少同名图片"))

    return samples, skipped


class ImageLabelDataset(Dataset):
    """images/labels 配对数据集。samples: [(img_path, label_idx), ...]。"""

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_path, label = self.samples[i]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_dataloaders():
    """读取 data/images + data/labels, 划分 train/val, 返回 (train_loader, val_loader, info)。"""
    samples, skipped = scan_pairs(settings.IMAGES_DIR, settings.LABELS_DIR)
    if not samples:
        raise RuntimeError(
            f"未在 {settings.IMAGES_DIR} / {settings.LABELS_DIR} 找到任何合法的"
            f" 图片-标签 配对。请检查数据布局(images/*.jpg 与 labels/*.txt 同名)。")
    if skipped:
        print(f"[warn] 跳过 {len(skipped)} 个异常项(前 5 个): {skipped[:5]}")

    # 划分 train/val(与原逻辑一致: 固定 seed 的 random_split)
    n_val = int(len(samples) * settings.VAL_RATIO)
    n_train = len(samples) - n_val
    g = torch.Generator().manual_seed(settings.SEED)
    train_idx, val_idx = random_split(range(len(samples)), [n_train, n_val], generator=g)

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]

    # train 带增强, val 不带(预处理与 test.py 一致)
    train_set = ImageLabelDataset(train_samples, get_transforms(train=True))
    val_set = ImageLabelDataset(val_samples, get_transforms(train=False))

    # 并行读盘提速: num_workers 多进程, pin_memory 加速到 GPU 的拷贝
    nw = settings.NUM_WORKERS
    pin = torch.cuda.is_available()
    common = dict(batch_size=settings.BATCH_SIZE, num_workers=nw,
                  pin_memory=pin, persistent_workers=(nw > 0))
    train_loader = DataLoader(train_set, shuffle=True, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)

    # 各类别样本数统计
    label_counts = {c: 0 for c in settings.CLASS_NAMES}
    for _, idx in samples:
        label_counts[settings.CLASS_NAMES[idx]] += 1

    info = {
        "total": len(samples), "n_train": n_train, "n_val": n_val,
        "label_counts": label_counts, "skipped": skipped,
        "val_samples": val_samples,   # 供 visualize.py 复用(拿原图路径)
    }
    return train_loader, val_loader, info
