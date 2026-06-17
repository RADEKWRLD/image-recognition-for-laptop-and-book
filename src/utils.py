# -*- coding: utf-8 -*-
"""
utils.py —— 通用工具函数。

只依赖 settings, 不依赖 model/dataset/trainer, 处于较底层。
"""

import os
import csv
import random

import numpy as np
import torch


def get_device():
    """选择计算设备: Mac 用 MPS, 有 N 卡用 CUDA, 否则 CPU。"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int):
    """固定随机种子, 保证实验可复现(报告里调参对比要可信)。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_label(name: str) -> str:
    """文件夹名 -> 标准下划线标签, 如 'laptop open' / 'Laptop_Open' -> 'laptop_open'。"""
    return name.strip().replace(" ", "_").lower()


def save_history_csv(history, path):
    """把每个 epoch 的指标存成 csv, 方便自己再画图/写报告。
    history: list of dict, 每个含 train_loss/train_acc/val_loss/val_acc。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for i, h in enumerate(history, 1):
            w.writerow([i, h["train_loss"], h["train_acc"],
                        h["val_loss"], h["val_acc"]])


def plot_curves(history, path):
    """画训练曲线(loss 和 acc 各一张), 报告第 2 部分需要 >=10 张曲线/分析。"""
    try:
        import matplotlib
        matplotlib.use("Agg")          # 无界面后端, 直接存图
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] 未安装 matplotlib, 跳过画图")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    eps = range(1, len(history) + 1)
    tl = [h["train_loss"] for h in history]
    ta = [h["train_acc"] for h in history]
    vl = [h["val_loss"] for h in history]
    va = [h["val_acc"] for h in history]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(eps, tl, label="train"); ax[0].plot(eps, vl, label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(eps, ta, label="train"); ax[1].plot(eps, va, label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("acc")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[info] 训练曲线已保存: {path}")
