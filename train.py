# -*- coding: utf-8 -*-
"""
train.py —— 训练程序入口(课前用)。

组装 src/ 各模块, 完成: 加载数据 -> 构建 CCT -> 训练 -> 保存权重/曲线/csv。
运行: python train.py
产物(runs/):
  best_model.pth  验证集最优权重 + 类别映射 + 预处理参数(test.py 加载它)
  curves.png      训练曲线(写报告用)
  history.csv     每轮指标
"""

import os

from src import settings
from src.utils import get_device, set_seed, plot_curves, save_history_csv
from src.dataset import build_dataloaders
from src.model import build_model, save_checkpoint
from src.trainer import Trainer


def main():
    set_seed(settings.SEED)
    os.makedirs(settings.OUT_DIR, exist_ok=True)
    device = get_device()
    print(f"[info] 设备: {device}")

    # 1. 数据
    train_loader, val_loader, info = build_dataloaders()
    print(f"[info] 各类别样本数: {info['label_counts']}")
    if info["skipped"]:
        print(f"[info] 跳过异常项 {len(info['skipped'])} 个")
    print(f"[info] 样本总数={info['total']} 训练={info['n_train']} 验证={info['n_val']}")

    # 2. 模型(精简版 CCT)
    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] 模型: 精简版 CCT, 参数量={n_params/1e6:.2f}M")

    # 3. 训练
    print("[info] 开始训练...")
    trainer = Trainer(model, device, lr=settings.LR, epochs=settings.EPOCHS)
    history, best_state, best_acc = trainer.fit(
        train_loader, val_loader, settings.EPOCHS)
    print(f"[info] 最优验证准确率 = {best_acc:.3f}")

    # 4. 保存产物(保存 best 权重)
    model.load_state_dict(best_state)
    save_checkpoint(model, settings.CKPT_PATH)
    print(f"[info] 权重已保存: {settings.CKPT_PATH}")
    save_history_csv(history, os.path.join(settings.OUT_DIR, "history.csv"))
    plot_curves(history, os.path.join(settings.OUT_DIR, "curves.png"))


if __name__ == "__main__":
    main()
