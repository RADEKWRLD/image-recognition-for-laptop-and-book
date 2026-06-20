# -*- coding: utf-8 -*-
"""
eda.py —— 数据探索分析(EDA)。

读 data/images + data/labels(同名配对), 对原图做统计与可视化, 产物存 runs/eda/:
  class_distribution.png  各类别数量柱状图
  image_sizes.png         宽x高散点 + 长宽比直方图
  color_stats.png         各类 RGB 通道均值 + 整体亮度直方图
  samples_grid.png        每类随机抽 N 张拼图
  eda_summary.csv         汇总表(总数/各类数量/尺寸统计/配对异常)

运行: python eda.py
只读不改原数据。
"""

import os
import csv

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")            # 无界面后端, 直接存图(同 utils.plot_curves)
import matplotlib.pyplot as plt

from src import settings
from src.utils import set_seed
from src.dataset import scan_pairs

SAMPLES_PER_CLASS = 6            # 样本网格每类抽几张


def collect_stats(samples):
    """逐张读原图, 收集尺寸/颜色统计。返回 per-sample 列表。"""
    rows = []
    for img_path, label_idx in samples:
        cls = settings.CLASS_NAMES[label_idx]
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                w, h = im.size
                arr = np.asarray(im, dtype=np.float32)      # (H,W,3)
                rgb_mean = arr.reshape(-1, 3).mean(axis=0)  # 每通道均值
                brightness = float(arr.mean())              # 整体亮度
            rows.append({
                "file": os.path.basename(img_path), "label": cls,
                "w": w, "h": h, "ratio": w / h,
                "r": rgb_mean[0], "g": rgb_mean[1], "b": rgb_mean[2],
                "brightness": brightness,
            })
        except Exception as e:
            rows.append({"file": os.path.basename(img_path), "label": cls,
                         "w": None, "h": None, "ratio": None,
                         "r": None, "g": None, "b": None, "brightness": None,
                         "error": str(e)})
    return rows


def plot_class_distribution(rows, out):
    counts = {c: 0 for c in settings.CLASS_NAMES}
    for r in rows:
        counts[r["label"]] += 1
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(counts.keys()), list(counts.values()), color="#4C72B0")
    for i, v in enumerate(counts.values()):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_title("Class distribution"); ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15); plt.tight_layout()
    plt.savefig(out, dpi=120); plt.close()
    return counts


def plot_image_sizes(rows, out):
    valid = [r for r in rows if r["w"]]
    ws = [r["w"] for r in valid]; hs = [r["h"] for r in valid]
    ratios = [r["ratio"] for r in valid]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(ws, hs, alpha=0.5, s=20, color="#C44E52")
    ax[0].set_title("Image size (W x H)"); ax[0].set_xlabel("width"); ax[0].set_ylabel("height")
    ax[0].grid(alpha=0.3)
    ax[1].hist(ratios, bins=20, color="#55A868")
    ax[1].axvline(1.0, color="k", ls="--", lw=1, label="square")
    ax[1].set_title("Aspect ratio (W/H)"); ax[1].set_xlabel("ratio"); ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()


def plot_color_stats(rows, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    # 各类 RGB 通道均值(分组柱状)
    classes = settings.CLASS_NAMES
    chans = {"r": [], "g": [], "b": []}
    for c in classes:
        sub = [r for r in rows if r["label"] == c and r["r"] is not None]
        for ch in ("r", "g", "b"):
            chans[ch].append(np.mean([r[ch] for r in sub]) if sub else 0)
    x = np.arange(len(classes)); width = 0.25
    ax[0].bar(x - width, chans["r"], width, label="R", color="#C44E52")
    ax[0].bar(x,         chans["g"], width, label="G", color="#55A868")
    ax[0].bar(x + width, chans["b"], width, label="B", color="#4C72B0")
    ax[0].set_xticks(x); ax[0].set_xticklabels(classes, rotation=15)
    ax[0].set_title("Mean RGB per class"); ax[0].set_ylabel("0-255"); ax[0].legend()
    ax[0].grid(axis="y", alpha=0.3)
    # 整体亮度直方图
    bright = [r["brightness"] for r in rows if r["brightness"] is not None]
    ax[1].hist(bright, bins=20, color="#8172B3")
    ax[1].set_title("Brightness distribution"); ax[1].set_xlabel("mean pixel (0-255)")
    ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()


def plot_samples_grid(samples, out, n_per_class=SAMPLES_PER_CLASS):
    set_seed(settings.SEED)
    by_class = {c: [] for c in settings.CLASS_NAMES}
    for path, idx in samples:
        by_class[settings.CLASS_NAMES[idx]].append(path)
    ncols = n_per_class; nrows = len(settings.CLASS_NAMES)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.7))
    axes = np.atleast_2d(axes)
    for r, cls in enumerate(settings.CLASS_NAMES):
        paths = by_class[cls]
        pick = list(np.random.choice(paths, min(n_per_class, len(paths)), replace=False)) if paths else []
        for c in range(ncols):
            ax = axes[r][c]; ax.axis("off")
            if c < len(pick):
                try:
                    ax.imshow(Image.open(pick[c]).convert("RGB"))
                except Exception:
                    ax.text(0.5, 0.5, "x", ha="center", va="center")
            if c == 0:
                ax.set_title(cls, loc="left", fontsize=9)
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()


def save_summary_csv(rows, counts, skipped, out):
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["==== 各类别数量 ===="])
        for c, n in counts.items():
            w.writerow([c, n])
        valid = [r for r in rows if r["w"]]
        if valid:
            ws = [r["w"] for r in valid]; hs = [r["h"] for r in valid]
            w.writerow([]); w.writerow(["==== 尺寸统计(W/H) ===="])
            w.writerow(["width_min", min(ws), "width_max", max(ws), "width_mean", round(np.mean(ws), 1)])
            w.writerow(["height_min", min(hs), "height_max", max(hs), "height_mean", round(np.mean(hs), 1)])
        w.writerow([]); w.writerow(["==== 配对异常 ===="])
        if skipped:
            for name, reason in skipped:
                w.writerow([name, reason])
        else:
            w.writerow(["(无)"])
        w.writerow([]); w.writerow(["==== 逐图明细 ===="])
        w.writerow(["file", "label", "w", "h", "ratio", "brightness"])
        for r in rows:
            w.writerow([r["file"], r["label"], r["w"], r["h"],
                        round(r["ratio"], 3) if r["ratio"] else None,
                        round(r["brightness"], 1) if r["brightness"] else None])


def main():
    out_dir = os.path.join(settings.OUT_DIR, "eda")
    os.makedirs(out_dir, exist_ok=True)

    samples, skipped = scan_pairs(settings.IMAGES_DIR, settings.LABELS_DIR)
    print(f"[info] images={settings.IMAGES_DIR}")
    print(f"[info] labels={settings.LABELS_DIR}")
    if not samples:
        print("[错误] 没找到任何合法的 图片-标签 配对, 请检查 data/images 与 data/labels")
        if skipped:
            print(f"[info] 异常项(前 10): {skipped[:10]}")
        return

    rows = collect_stats(samples)
    counts = plot_class_distribution(rows, os.path.join(out_dir, "class_distribution.png"))
    plot_image_sizes(rows, os.path.join(out_dir, "image_sizes.png"))
    plot_color_stats(rows, os.path.join(out_dir, "color_stats.png"))
    plot_samples_grid(samples, os.path.join(out_dir, "samples_grid.png"))
    save_summary_csv(rows, counts, skipped, os.path.join(out_dir, "eda_summary.csv"))

    # 终端汇总
    print(f"[info] 合法样本 {len(samples)} 张, 各类别: {counts}")
    bad = [r for r in rows if r["w"] is None]
    print(f"[info] 损坏/无法读取: {len(bad)} 张")
    print(f"[info] 配对异常: {len(skipped)} 个" + (f" -> {skipped[:5]}" if skipped else ""))
    print(f"[info] EDA 产物已保存到: {out_dir}")


if __name__ == "__main__":
    main()
