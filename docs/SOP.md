# 启动 SOP（训练 + 测试全流程）

本文档给出从零跑通本项目的标准操作流程，覆盖：**环境安装 → 准备数据 → 训练 → 测试 → 校验**。
命令均可直接照抄。项目根目录为 `image-recognition/`。

> 模型与各层算法见 [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)。

---

## 步骤 0｜环境准备

需要 Python 3.13。安装依赖（版本已在 `requirements.txt` 中定死）：

```bash
pip install -r requirements.txt
```

设备说明：
- **Apple Silicon（M 系列）** 自动使用 MPS，无需改代码。
- 有 NVIDIA 显卡自动用 CUDA；否则回退 CPU。优先级 **MPS > CUDA > CPU**。


---

## 步骤 1｜准备数据

在根目录 `data/` 下放 **4 个类别子目录**，每个目录放该类别图片：

```
data/
├── book_close/      # 合着的书
├── book_open/       # 打开的书
├── laptop_close/    # 合着的笔记本
└── laptop_open/     # 打开的笔记本
```

说明：
- 文件夹名会经 `normalize_label()` 自动归一化（去空格、小写、下划线化），如 `Laptop Open` → `laptop_open`。
- 图片格式支持 `.jpg/.jpeg/.png/.bmp/.webp`。
- 训练时按 `VAL_RATIO=0.2` 自动 80/20 划分训练/验证集。

---

## 步骤 2｜（可选）调参

无需调参可跳过。要改超参，编辑根目录 `.env`：

| 参数 | 默认 | 影响 |
|---|---|---|
| `IMG_SIZE` | 96 | ↑ 保留更多细节，但计算量增大 |
| `BATCH_SIZE` | 16 | ↑ 收敛更快，占用显存更多 |
| `EPOCHS` | 40 | ↑ 收敛更充分，但过拟合风险增大 |
| `LR` | 5e-4 | ↑ 收敛快但不稳；↓ 稳但慢 |
| `VAL_RATIO` | 0.2 | 验证集比例 |
| `EMBED_DIM` | 128 | Transformer 维度，↑ 表达更强但易过拟合 |
| `NUM_LAYERS` / `NUM_HEADS` | 2 / 4 | Transformer 深度/注意力头数 |
| `DROPOUT` | 0.1 | ↑ 抗过拟合更强 |

> ⚠️ 不要改动 `CLASS_NAMES` 的顺序（在 `src/settings.py`，故意不放进 `.env`）——顺序即模型输出 index，改乱会导致标签错位。

---

## 步骤 3｜训练

```bash
python train.py
```

产物（生成在 `runs/`）：

| 文件 | 说明 |
|---|---|
| `best_model.pth` | 验证集最优权重 + 类别映射 + 预处理参数（`test.py` 加载它） |
| `curves.png` | 训练/验证 Loss 与 Accuracy 曲线（写报告用） |
| `history.csv` | 每轮 `train_loss / train_acc / val_loss / val_acc` |

看曲线判断：
- **正常收敛**：train/val 的 loss 都下降、acc 都上升。
- **过拟合**：train_acc 很高但 val_acc 停滞或下降 → 增大 `DROPOUT`、减小 `EPOCHS`，或补充数据。

---

## 步骤 4｜测试 / 推理（现场）

```bash
python test.py <测试图片文件夹> [输出文件夹]
```

示例（不带参数时默认 `test_images` → `test_images/output`）：

```bash
python test.py test_images test_images/output
```

行为：
- 遍历输入文件夹所有图片，逐张 Resize→Normalize→`argmax`，给每张图输出一个**同名 `.txt`**，内容为 `book_close / book_open / laptop_close / laptop_open` 之一。
- 仅加载权重、不训练，**要求启动后 1 分钟内完成全部推理**；运行结束会打印总耗时与 ms/张，并提示是否满足 1 分钟。

---

## 步骤 5｜结果校验

仅检查输出格式是否合法：

```bash
python format_check.py test_images/output
```

对照标签计算准确率（需准备 `labels.csv`）：

```bash
python format_check.py test_images/output labels.csv
```

校验内容：txt 文件名/内容是否为合法标签，避免格式错误；提供对照表时输出准确率。

---

## 附｜常见问题（FAQ）

| 现象 | 原因 / 处理 |
|---|---|
| `test.py` 报找不到 `best_model.pth` | 还没训练 → 先跑 `python train.py` |
| 某些图片没被处理 | 扩展名不在 `.jpg/.jpeg/.png/.bmp/.webp` 内，转换格式后重试 |
| 推理超过 1 分钟 | 减小 `IMG_SIZE`，或确认用了 MPS/CUDA 而非 CPU |
| 预测标签整体错位 | 不要改 `CLASS_NAMES` 顺序；用同一份代码训练与推理 |
| 训练/测试结果不一致 | 预处理由 checkpoint 统一保证，确保 `test.py` 加载的是本次训练产出的权重 |

---

## 一页速查

```bash
# 1. 装依赖
pip install -r requirements.txt
# 2. 放数据到 data/{book_close,book_open,laptop_close,laptop_open}/
# 3. 训练（产出 runs/best_model.pth）
python train.py
# 4. 推理（产出每图同名 .txt）
python test.py test_images test_images/output
# 5. 校验
python format_check.py test_images/output            # 仅格式
python format_check.py test_images/output labels.csv # 带精度
```
