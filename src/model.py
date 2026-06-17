# -*- coding: utf-8 -*-
"""
model.py —— 模型架构: 精简版 CCT (Compact Convolutional Transformer)。

设计依据论文: Hassani et al., "Escaping the Big Data Paradigm with Compact
Transformers" (2021, arXiv:2104.05704)。核心思想是让 Transformer 也能在小数据上
从零训练:
  1) 卷积 tokenizer —— 用卷积代替 ViT 的硬切块, 注入 CNN 的归纳偏置(局部性、
     平移等变), 这是小数据不过拟合的关键;
  2) 轻量 Transformer encoder —— 建模 token 之间的全局关系;
  3) SeqPool(序列池化)—— 用注意力把 N 个 token 聚合成 1 个向量, 替代 class
     token, 更省参也更准。

结构(默认 IMG_SIZE=96):
  图 96x96x3 --卷积tokenizer(下采样8倍)--> 12x12=144 个 token, 每个 EMBED_DIM 维
            --加位置编码--> TransformerEncoder x NUM_LAYERS
            --SeqPool--> 一个向量 --Linear--> 4 类
"""

import torch
import torch.nn as nn

from src import settings


class ConvTokenizer(nn.Module):
    """卷积式 tokenizer: 3 个 (Conv-BN-ReLU-MaxPool) 块, 把图像变成 token 序列。
    每个块把空间尺寸减半, 共下采样 8 倍。输出形状 (B, N, embed_dim)。
    """
    def __init__(self, in_ch=3, embed_dim=128):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # /2
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # /4
            nn.Conv2d(64, embed_dim, 3, 1, 1), nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # /8
        )

    def forward(self, x):
        x = self.block(x)                    # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)     # (B, N, D), N = H'*W'
        return x


class SeqPool(nn.Module):
    """序列池化(CCT 关键设计): 给每个 token 学一个注意力权重, 加权求和成单向量。"""
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.Linear(embed_dim, 1)

    def forward(self, x):                    # x: (B, N, D)
        w = torch.softmax(self.attn(x), dim=1)   # (B, N, 1), 在 token 维归一化
        return (w * x).sum(dim=1)            # (B, D)


class CCT(nn.Module):
    """精简版 Compact Convolutional Transformer。"""
    def __init__(self, img_size, in_ch, num_classes,
                 embed_dim, num_layers, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.tokenizer = ConvTokenizer(in_ch, embed_dim)

        # 用一个假输入推断 token 数量 N(随 img_size 变化)
        with torch.no_grad():
            n_tokens = self.tokenizer(torch.zeros(1, in_ch, img_size, img_size)).shape[1]

        # 可学习位置编码(让 Transformer 知道 token 的空间位置)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)
        self.seqpool = SeqPool(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.tokenizer(x)                # (B, N, D)
        x = self.drop(x + self.pos_emb)      # 加位置编码
        x = self.encoder(x)                  # 自注意力建模全局关系
        x = self.norm(x)
        x = self.seqpool(x)                  # (B, D)
        return self.head(x)                  # (B, num_classes)


def build_model():
    """按 settings 里的超参构建一个新的 CCT(随机初始化, 用于训练)。"""
    return CCT(
        img_size=settings.IMG_SIZE, in_ch=3, num_classes=settings.NUM_CLASSES,
        embed_dim=settings.EMBED_DIM, num_layers=settings.NUM_LAYERS,
        num_heads=settings.NUM_HEADS, mlp_ratio=settings.MLP_RATIO,
        dropout=settings.DROPOUT)


def save_checkpoint(model, path):
    """保存权重 + 重建模型所需的全部信息 + 预处理参数。
    把 img_size/mean/std/class_names 一并存入, 保证 test.py 的预处理与训练完全一致。
    """
    torch.save({
        "state_dict": model.state_dict(),
        "class_names": settings.CLASS_NAMES,
        "img_size": settings.IMG_SIZE,
        "mean": settings.MEAN, "std": settings.STD,
        # 重建结构所需超参
        "arch": dict(embed_dim=settings.EMBED_DIM, num_layers=settings.NUM_LAYERS,
                     num_heads=settings.NUM_HEADS, mlp_ratio=settings.MLP_RATIO,
                     dropout=settings.DROPOUT),
    }, path)


def load_model(path, device):
    """从权重文件重建模型并载入参数(test.py 用)。返回 (model, ckpt 元信息)。"""
    ckpt = torch.load(path, map_location=device)
    a = ckpt["arch"]
    model = CCT(img_size=ckpt["img_size"], in_ch=3,
                num_classes=len(ckpt["class_names"]),
                embed_dim=a["embed_dim"], num_layers=a["num_layers"],
                num_heads=a["num_heads"], mlp_ratio=a["mlp_ratio"],
                dropout=a["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
