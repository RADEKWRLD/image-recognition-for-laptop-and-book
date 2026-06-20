# -*- coding: utf-8 -*-
"""
visualize.py —— 预测结果可视化。

用训练好的权重(runs/best_model.pth)对一批图片推理, 把「错在哪、为什么错」画出来。
产物存 runs/viz/:
  confusion_matrix.png  4x4 混淆矩阵热力图(纯 numpy, 不引入 sklearn)
  per_class_acc.png     每类准确率柱状图
  predictions_grid.png  抽样图标注 真值/预测(置信度), 正确绿/错误红
  misclassified.png     错分样本画廊, 标 真->预测
  confidence_hist.png   正确 vs 错误样本的 softmax 最大概率直方图

模式 A(默认): python visualize.py
    用验证集(data/images+labels 划分出的 val)推理, 自带真值。
模式 B: python visualize.py <图片文件夹> <labels文件夹>
    对外部 images+labels 推理(真值来自同名 txt)。
"""

import os
import sys

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import settings
from src.utils import get_device
from src.model import load_model
from src.dataset import get_transforms, build_dataloaders, scan_pairs

GRID_N = 12                 # 预测网格抽样张数
MISCLS_N = 12              # 错分画廊最多展示张数


def gather_samples(argv):
    """根据命令行参数返回 [(img_path, true_idx), ...]。"""
    if len(argv) >= 3:                       # 模式 B: 外部 images + labels
        samples, skipped = scan_pairs(argv[1], argv[2])
        print(f"[info] 模式B 外部数据: {argv[1]} / {argv[2]}")
        if skipped:
            print(f"[warn] 跳过 {len(skipped)} 个异常项")
        return samples
    # 模式 A: 复用 build_dataloaders 的验证集
    _, _, info = build_dataloaders()
    print(f"[info] 模式A 验证集 {len(info['val_samples'])} 张")
    return info["val_samples"]


@torch.no_grad()
def predict(model, device, samples):
    """逐张推理, 返回 (y_true, y_pred, confs, paths)。"""
    tf = get_transforms(train=False)
    y_true, y_pred, confs, paths = [], [], [], []
    for img_path, true_idx in samples:
        img = Image.open(img_path).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        prob = torch.softmax(model(x), dim=1)[0]
        conf, pred = prob.max(0)
        y_true.append(true_idx); y_pred.append(int(pred))
        confs.append(float(conf)); paths.append(img_path)
    return np.array(y_true), np.array(y_pred), np.array(confs), paths


def plot_confusion(y_true, y_pred, out):
    n = len(settings.CLASS_NAMES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    acc = (y_true == y_pred).mean() if len(y_true) else 0.0
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(settings.CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(settings.CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix (acc={acc:.3f})")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    return cm, acc


def plot_per_class_acc(cm, out):
    n = len(settings.CLASS_NAMES)
    accs = [cm[i, i] / cm[i].sum() if cm[i].sum() else 0.0 for i in range(n)]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(settings.CLASS_NAMES, accs, color="#4C72B0")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a, f"{a:.2f}", ha="center", va="bottom")
    ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy"); ax.set_title("Per-class accuracy")
    plt.xticks(rotation=15); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    return accs


def _img_grid(items, out, title):
    """items: [(path, true_idx, pred_idx, conf), ...]。正确绿/错误红标题。"""
    if not items:
        return
    ncols = 4; nrows = (len(items) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.4, nrows * 2.6))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (path, t, p, conf) in zip(axes, items):
        try:
            ax.imshow(Image.open(path).convert("RGB"))
        except Exception:
            ax.text(0.5, 0.5, "x", ha="center", va="center")
        ok = (t == p)
        ax.set_title(f"T:{settings.CLASS_NAMES[t]}\nP:{settings.CLASS_NAMES[p]} ({conf:.2f})",
                     fontsize=8, color="green" if ok else "red")
    fig.suptitle(title)
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()


def plot_predictions_grid(y_true, y_pred, confs, paths, out, n=GRID_N):
    idx = list(range(len(paths)))[:n]
    items = [(paths[i], int(y_true[i]), int(y_pred[i]), confs[i]) for i in idx]
    _img_grid(items, out, "Predictions (T=true, P=pred)")


def plot_misclassified(y_true, y_pred, confs, paths, out, n=MISCLS_N):
    wrong = [i for i in range(len(paths)) if y_true[i] != y_pred[i]][:n]
    items = [(paths[i], int(y_true[i]), int(y_pred[i]), confs[i]) for i in wrong]
    _img_grid(items, out, "Misclassified (T->P)")
    return len(wrong)


def plot_confidence_hist(y_true, y_pred, confs, out):
    correct = confs[y_true == y_pred]
    wrong = confs[y_true != y_pred]
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0.25, 1.0, 16)
    ax.hist(correct, bins=bins, alpha=0.6, label=f"correct ({len(correct)})", color="#55A868")
    ax.hist(wrong, bins=bins, alpha=0.6, label=f"wrong ({len(wrong)})", color="#C44E52")
    ax.set_xlabel("max softmax prob"); ax.set_ylabel("count")
    ax.set_title("Confidence: correct vs wrong"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()


def main():
    if not os.path.exists(settings.CKPT_PATH):
        print(f"[错误] 找不到权重 {settings.CKPT_PATH}, 请先运行 python train.py")
        return
    out_dir = os.path.join(settings.OUT_DIR, "viz")
    os.makedirs(out_dir, exist_ok=True)

    device = get_device()
    model, _ = load_model(settings.CKPT_PATH, device)
    print(f"[info] 设备={device}  权重={settings.CKPT_PATH}")

    samples = gather_samples(sys.argv)
    if not samples:
        print("[错误] 没有可推理的样本, 检查输入数据")
        return

    y_true, y_pred, confs, paths = predict(model, device, samples)

    cm, acc = plot_confusion(y_true, y_pred, os.path.join(out_dir, "confusion_matrix.png"))
    accs = plot_per_class_acc(cm, os.path.join(out_dir, "per_class_acc.png"))
    plot_predictions_grid(y_true, y_pred, confs, paths, os.path.join(out_dir, "predictions_grid.png"))
    n_wrong = plot_misclassified(y_true, y_pred, confs, paths, os.path.join(out_dir, "misclassified.png"))
    plot_confidence_hist(y_true, y_pred, confs, os.path.join(out_dir, "confidence_hist.png"))

    # 终端汇总
    print(f"[info] 整体准确率 = {acc:.3f}  ({(y_true == y_pred).sum()}/{len(y_true)})")
    for c, a in zip(settings.CLASS_NAMES, accs):
        print(f"       {c:14s} acc={a:.3f}")
    print(f"[info] 错分 {n_wrong} 张")
    print(f"[info] 可视化产物已保存到: {out_dir}")


if __name__ == "__main__":
    main()
