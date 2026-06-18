# 架构说明（Architecture）

本项目是一个 **4 分类图像识别系统**，类别为 `book_close / book_open / laptop_close / laptop_open`。
核心模型为**精简版 CCT（Compact Convolutional Transformer）**，依据论文
Hassani et al., *"Escaping the Big Data Paradigm with Compact Transformers"* (2021, arXiv:2104.05704)，
专为**小数据集从零训练**设计。

代码采用单向依赖的分层结构（下层不依赖上层），全部位于 `src/`，由根目录的入口脚本组装。

---

## 一、整体数据流

```
                .env(超参/路径)
                   │
                   ▼
          ┌─────────────────┐
          │ L0  settings.py │  配置加载 + 固定常量
          └────────┬────────┘
        ┌──────────┼───────────┐
        ▼          ▼           ▼
   L1 utils.py  L2 model.py  L2 dataset.py
   (工具)        (CCT 网络)   (数据+预处理)
        └──────────┬───────────┘
                   ▼
            L3 trainer.py（训练循环）
                   │
        ┌──────────┴───────────┐
        ▼                      ▼
   train.py(训练入口)     test.py(推理入口)
        │                      │
   runs/best_model.pth ───────►│ 加载权重 → 推理 → 输出 .txt
   runs/curves.png             │
   runs/history.csv            ▼
                          format_check.py（输出校验/算精度）
```

---

## 二、分层职责与算法

### L0 配置层 — [`src/settings.py`](../src/settings.py)

| 项 | 内容 |
|---|---|
| **输入** | 根目录 `.env` + 代码内固定常量 |
| **算法/技术** | `python-dotenv` 加载 `.env`，按类型转换（int/float/str）；统一对外暴露配置，其它模块不直接 `os.getenv` |
| **输出** | 全局配置变量 |
| **职责** | 中央配置管理；固定 `CLASS_NAMES` 顺序（即模型输出 index 顺序，防标签错位）；定义 ImageNet 归一化 `MEAN/STD` |

关键超参（默认值）：

- 训练：`IMG_SIZE=96`、`BATCH_SIZE=16`、`EPOCHS=40`、`LR=5e-4`、`VAL_RATIO=0.2`、`SEED=42`
- CCT 结构：`EMBED_DIM=128`、`NUM_LAYERS=2`、`NUM_HEADS=4`、`MLP_RATIO=2.0`、`DROPOUT=0.1`
- 固定常量：`CLASS_NAMES=["book_close","book_open","laptop_close","laptop_open"]`、`MEAN=[0.485,0.456,0.406]`、`STD=[0.229,0.224,0.225]`

---

### L1 工具层 — [`src/utils.py`](../src/utils.py)

| 项 | 内容 |
|---|---|
| **输入** | 配置 / 训练历史 |
| **算法/技术** | `get_device()` 设备选择优先级 **MPS > CUDA > CPU**；`set_seed()` 固定 random/numpy/torch 种子（可复现）；`normalize_label()` 文件夹名归一化（去空格、小写、下划线化）；`save_history_csv()`、`plot_curves()`（matplotlib 画 Loss/Accuracy 曲线） |
| **输出** | 设备对象、工具函数、csv/png 文件 |
| **职责** | 跨模块通用能力 |

---

### L2 数据层 — [`src/dataset.py`](../src/dataset.py)

| 项 | 内容 |
|---|---|
| **输入** | `data/` 下 4 个类别子目录 |
| **算法/技术** | `torchvision.ImageFolder` 读取；`get_transforms(train)`：<br>· **训练**：Resize(96) + **数据增强**（RandomHorizontalFlip、RandomRotation(15°)、ColorJitter(亮度/对比度/饱和度 0.2)）+ ToTensor + Normalize<br>· **验证/测试**：仅 Resize(96) + ToTensor + Normalize（**无增强**，保证 train/test 预处理一致）<br>按 `VAL_RATIO=0.2` 做 80/20 划分，train/val 各绑定对应 transform |
| **输出** | `train_loader`、`val_loader`、info 字典 |
| **职责** | 数据批处理与标签映射；数据增强缓解小数据过拟合 |

---

### L2 模型层 — [`src/model.py`](../src/model.py) ★核心

**前向推理流水（默认 `IMG_SIZE=96`）：**

```
输入 96×96×3 RGB
   │
   ▼ ① ConvTokenizer（下采样 8 倍）
   │    Conv3×3(3→32)  + BN + ReLU + MaxPool2   → 48×48
   │    Conv3×3(32→64) + BN + ReLU + MaxPool2   → 24×24
   │    Conv3×3(64→128)+ BN + ReLU + MaxPool2   → 12×12
   │    flatten+transpose → 144 个 token, 每个 128 维   (B,144,128)
   ▼ ② + 可学习位置编码 pos_emb (trunc_normal 初始化) + Dropout
   ▼ ③ TransformerEncoder × 2 层
   │    多头自注意力 4 头；FFN 维度 128×2=256；GELU；Dropout=0.1
   ▼ ④ LayerNorm
   ▼ ⑤ SeqPool（CCT 关键创新）
   │    每个 token 学注意力权重 Linear(128→1)→softmax，加权求和 → (B,128)
   ▼ ⑥ Linear Head 128 → 4
   │
输出 4 类 logits（softmax 后为概率）
```

逐子模块算法：

| 子模块 | 算法 | 作用 |
|---|---|---|
| **ConvTokenizer** | 3×（Conv3×3 + BatchNorm + ReLU + MaxPool2） | 用卷积代替 ViT 硬切块，注入 CNN 局部性/平移等变归纳偏置——小数据不过拟合的关键 |
| **位置编码** | 可学习 `pos_emb`（trunc_normal, std=0.02）+ Dropout | 让 Transformer 知道每个 token 的空间位置 |
| **TransformerEncoder** | `nn.TransformerEncoderLayer`，2 层 / 4 头 / FFN=256 / GELU | 多头自注意力建模 token 间全局关系 |
| **LayerNorm** | 层归一化 | 稳定训练 |
| **SeqPool** | 注意力加权池化（Linear→softmax→加权和） | 替代 class token，把 144 个 token 聚合成 1 个向量，更省参更稳 |
| **Head** | `Linear(128→4)` | 输出 4 类 logits |

**模型持久化**

- `save_checkpoint()`：保存 `state_dict + class_names + img_size + mean/std + arch 超参`，保证 `test.py` 重建结构与预处理**完全一致**。
- `load_model()`：从 checkpoint 反向重建 CCT 并载入权重，置 `eval()`。

---

### L3 训练层 — [`src/trainer.py`](../src/trainer.py)

| 项 | 内容 |
|---|---|
| **输入** | 模型 + `train_loader` / `val_loader` |
| **算法/技术** | 损失 `CrossEntropyLoss(label_smoothing=0.1)`（标签平滑，防小数据过度自信）；优化器 `AdamW(lr=5e-4, weight_decay=1e-4)`（含 L2 正则）；学习率调度 `CosineAnnealingLR`（余弦退火，平滑收敛）；按验证集最优保留权重（early-stopping 思想） |
| **输出** | 训练历史 `history`、最优权重 `best_state`、最优验证精度 `best_acc` |
| **职责** | 参数优化与最优模型选择 |

---

### 入口层

| 文件 | 算法/职责 |
|---|---|
| [`train.py`](../train.py) | 组装全流程：set_seed → build_dataloaders → build_model → Trainer.fit → save_checkpoint + plot_curves + save_history_csv |
| [`test.py`](../test.py) | 仅加载权重做推理：load_model → 逐图 Resize/Normalize → `argmax` → 输出同名 `.txt`。要求启动后 **1 分钟内**完成全部推理 |
| [`format_check.py`](../format_check.py) | 校验输出 txt 文件名/内容是否为合法标签；可对照 `labels.csv` 计算准确率 |

---

## 三、层级总表

| 层 | 文件 | 输入 | 核心算法 | 输出 |
|---|---|---|---|---|
| L0 | `settings.py` | `.env` + 常量 | dotenv 加载 + 类型转换 | 配置变量 |
| L1 | `utils.py` | 配置 | 设备选择 / 种子 / 画图 | 工具函数 |
| L2 | `dataset.py` | `data/` | ImageFolder + 数据增强 + 80/20 划分 | DataLoader×2 |
| L2 | `model.py` | 配置 | ConvTokenizer + Transformer + SeqPool | CCT 网络 |
| L3 | `trainer.py` | DataLoader + 模型 | CrossEntropy(LS) + AdamW + 余弦退火 | best 权重 + history |
| 入口 | `train.py` / `test.py` | L0–L3 | 流程组装 / 推理 | 权重产物 / `.txt` 预测 |

---

## 四、关键设计点（为什么这样选）

1. **ConvTokenizer 注入归纳偏置** — 卷积带来局部性，让 Transformer 在 91 张小数据上也能从零训练。
2. **SeqPool 替代 class token** — 注意力加权池化，省参且更稳。
3. **标签平滑 + 数据增强 + 权重衰减** — 三重抗过拟合，针对小数据集。
4. **余弦退火学习率** — 平滑收敛。
5. **checkpoint 自带预处理参数** — 训练与测试预处理强一致，避免现场踩坑。
