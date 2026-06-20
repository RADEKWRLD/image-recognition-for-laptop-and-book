# -*- coding: utf-8 -*-
"""
trainer.py —— 训练逻辑。

封装训练/验证循环, 记录每个 epoch 的指标, 保留验证集最优权重。
CCT 是从零训练(无冻结骨干), 用 AdamW + 权重衰减 + 余弦退火学习率。
"""

import copy

import torch
import torch.nn as nn

try:
    from tqdm import tqdm
except ImportError:                  # 未装 tqdm 时退化为普通迭代, 不报错
    def tqdm(it, **kw):
        return it


class Trainer:
    def __init__(self, model, device, lr, epochs):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑, 小数据更稳
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        # 余弦退火: 学习率随训练平滑下降, 收敛更好
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs)

    def _run_epoch(self, loader, train: bool, desc=""):
        """跑一个 epoch, 返回 (平均loss, 准确率)。带 tqdm 进度条实时显示 loss/acc。"""
        self.model.train() if train else self.model.eval()
        tot_loss, tot_correct, tot = 0.0, 0, 0
        bar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        with torch.set_grad_enabled(train):
            for x, y in bar:
                x, y = x.to(self.device), y.to(self.device)
                if train:
                    self.optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y)
                if train:
                    loss.backward()
                    self.optimizer.step()
                tot_loss += loss.item() * x.size(0)
                tot_correct += (out.argmax(1) == y).sum().item()
                tot += x.size(0)
                if hasattr(bar, "set_postfix"):
                    bar.set_postfix(loss=f"{tot_loss/tot:.4f}", acc=f"{tot_correct/tot:.3f}")
        return tot_loss / tot, tot_correct / tot

    def fit(self, train_loader, val_loader, epochs):
        """训练 epochs 轮, 返回 (history, best_state, best_acc)。"""
        history = []
        best_acc, best_state = 0.0, copy.deepcopy(self.model.state_dict())
        for ep in range(1, epochs + 1):
            tr_loss, tr_acc = self._run_epoch(
                train_loader, train=True, desc=f"Epoch {ep:02d}/{epochs} [train]")
            va_loss, va_acc = self._run_epoch(
                val_loader, train=False, desc=f"Epoch {ep:02d}/{epochs} [val]")
            self.scheduler.step()
            history.append({"train_loss": tr_loss, "train_acc": tr_acc,
                            "val_loss": va_loss, "val_acc": va_acc})
            print(f"  Epoch {ep:02d}/{epochs} | "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.3f}")
            if va_acc >= best_acc:                       # 保留验证集最优
                best_acc = va_acc
                best_state = copy.deepcopy(self.model.state_dict())
        return history, best_state, best_acc
