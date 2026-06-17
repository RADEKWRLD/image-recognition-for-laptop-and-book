# -*- coding: utf-8 -*-
"""
src —— 4 分类图像识别项目的核心模块包。

分层结构(依赖单向, 不循环):
    settings.py  配置加载(读 .env + 固定常量), 最底层
    utils.py     通用工具(设备/种子/标签映射/画曲线)
    model.py     模型架构(精简版 CCT: 卷积 tokenizer + 轻量 Transformer + SeqPool)
    dataset.py   数据加载(transform + ImageFolder + 划分)
    trainer.py   训练逻辑(训练/验证循环, 保存最优)
"""
