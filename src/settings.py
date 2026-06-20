# -*- coding: utf-8 -*-
"""
settings.py —— 配置加载层(最底层模块)。

职责:
  1. load_dotenv() 读取项目根目录的 .env
  2. 把字符串型的 env 值转成正确类型(int/float)
  3. 暴露不该进 .env 的固定常量(CLASS_NAMES、归一化参数)

其它模块统一从这里 import 配置, 不直接调用 os.getenv。
"""

import os
from dotenv import load_dotenv

# 加载项目根目录的 .env(本文件在 src/ 下, 上一级即根目录)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"))


def _get(key, default, cast):
    """读取 env 并转型; 缺省时用 default。"""
    val = os.getenv(key)
    return cast(val) if val is not None else default


# ============ 路径(转成绝对路径, 不受运行目录影响) ============
DATA_DIR = os.path.join(_ROOT, _get("DATA_DIR", "data", str))
OUT_DIR  = os.path.join(_ROOT, _get("OUT_DIR", "runs", str))

# 数据采用 images/labels 同名配对布局: data/images/0001.jpg <-> data/labels/0001.txt
# (txt 内容为类别字符串)。子目录名可被 .env 覆盖。
IMAGES_DIR = os.path.join(DATA_DIR, _get("IMAGES_SUBDIR", "images", str))
LABELS_DIR = os.path.join(DATA_DIR, _get("LABELS_SUBDIR", "labels", str))

# ============ 训练超参 ============
IMG_SIZE   = _get("IMG_SIZE", 96, int)
BATCH_SIZE = _get("BATCH_SIZE", 16, int)
NUM_WORKERS = _get("NUM_WORKERS", 4, int)   # DataLoader 读盘并行度(CPU 核少可调小/设 0)
EPOCHS     = _get("EPOCHS", 40, int)
LR         = _get("LR", 5e-4, float)
VAL_RATIO  = _get("VAL_RATIO", 0.2, float)
SEED       = _get("SEED", 42, int)

# ============ CCT 结构超参 ============
EMBED_DIM  = _get("EMBED_DIM", 128, int)
NUM_LAYERS = _get("NUM_LAYERS", 2, int)
NUM_HEADS  = _get("NUM_HEADS", 4, int)
MLP_RATIO  = _get("MLP_RATIO", 2.0, float)
DROPOUT    = _get("DROPOUT", 0.1, float)

# ============ 固定常量(不进 .env, 防止顺序被改乱导致标签错位) ============
# 4 个类别, 顺序即模型输出 index 的顺序; 测试输出的 txt 内容必须是这几个字符串之一
CLASS_NAMES = ["book_close", "book_open", "laptop_close", "laptop_open"]
NUM_CLASSES = len(CLASS_NAMES)

# ImageNet 归一化参数(通用, 数据增强/预处理用)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# 权重文件路径
CKPT_PATH = os.path.join(OUT_DIR, "best_model.pth")
