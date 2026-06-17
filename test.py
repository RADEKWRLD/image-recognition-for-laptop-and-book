# -*- coding: utf-8 -*-
"""
test.py —— 测试程序入口(现场用)。

读入一个只含图片的文件夹, 用训练好的模型推理, 给每张图输出一个同名 .txt,
内容为 book_close / book_open / laptop_close / laptop_open 中的一种。
要求启动后 1 分钟内完成全部推理(故只加载权重、不训练)。

运行:
  python test.py <测试图片文件夹> [输出文件夹]
  例: python test.py test_images test_images/output
不给参数时默认 测试图=test_images, 输出=test_images/output。
"""

import os
import sys
import time

import torch
from PIL import Image

from src import settings
from src.utils import get_device
from src.model import load_model
from src.dataset import get_transforms

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    in_dir = sys.argv[1] if len(sys.argv) > 1 else "test_images"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(in_dir, "output")
    os.makedirs(out_dir, exist_ok=True)

    device = get_device()
    model, ckpt = load_model(settings.CKPT_PATH, device)
    class_names = ckpt["class_names"]
    tf = get_transforms(train=False)        # 与训练验证集完全一致的预处理
    print(f"[info] 设备={device}  权重={settings.CKPT_PATH}")

    # 收集图片(忽略已有的 output 子目录)
    files = sorted(f for f in os.listdir(in_dir)
                   if f.lower().endswith(IMG_EXT))
    print(f"[info] 待测图片 {len(files)} 张, 开始推理...")

    t0 = time.time()
    with torch.no_grad():
        for fn in files:
            img = Image.open(os.path.join(in_dir, fn)).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)          # (1,3,H,W)
            pred = model(x).argmax(1).item()
            label = class_names[pred]
            # 输出同名 txt(扩展名换成 .txt), 内容为标签字符串
            stem = os.path.splitext(fn)[0]
            with open(os.path.join(out_dir, stem + ".txt"), "w") as f:
                f.write(label)
    dt = time.time() - t0
    print(f"[info] 完成! {len(files)} 张耗时 {dt:.2f}s, 结果在: {out_dir}")
    print(f"[info] 平均 {dt/max(len(files),1)*1000:.1f} ms/张"
          f"  ({'✓ 满足1分钟' if dt < 60 else '✗ 超过1分钟, 需优化'})")


if __name__ == "__main__":
    main()
