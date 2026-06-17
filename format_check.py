# -*- coding: utf-8 -*-
"""
format_check.py —— 提交前自查脚本(对应作业里老师下发的 test.py)。

两个用途:
  1. 只查格式: 检查 output/ 里的 txt 文件名、内容是否合规(防止格式错直接 0 分)
       python format_check.py <txt文件夹>
  2. 查格式 + 算准确率: 再给一份"图名->真值标签"的 csv(两列: filename,label),
     就能算出准确率, 用于课前自测模型效果。
       python format_check.py <txt文件夹> <labels.csv>
"""

import os
import sys
import csv

VALID = {"book_close", "book_open", "laptop_close", "laptop_open"}


def check_format(txt_dir):
    """检查每个 txt 内容是否为合法标签, 返回 {stem: label}。"""
    preds, errors = {}, []
    txts = [f for f in os.listdir(txt_dir) if f.lower().endswith(".txt")]
    if not txts:
        print(f"[错误] {txt_dir} 里没有 txt 文件")
        return preds
    for fn in txts:
        with open(os.path.join(txt_dir, fn)) as f:
            content = f.read().strip()
        stem = os.path.splitext(fn)[0]
        if content not in VALID:
            errors.append(f"  {fn}: 内容 '{content}' 不是合法标签")
        else:
            preds[stem] = content
    print(f"[格式检查] 共 {len(txts)} 个 txt, 合法 {len(preds)} 个, 错误 {len(errors)} 个")
    for e in errors:
        print(e)
    if not errors:
        print("[✓] 全部格式合规, 可以打包上传")
    return preds


def check_accuracy(preds, labels_csv):
    """对照真值 csv 算准确率。csv 两列: filename(可带或不带扩展名), label。"""
    truth = {}
    with open(labels_csv) as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            stem = os.path.splitext(row[0].strip())[0]
            truth[stem] = row[1].strip()
    correct = sum(1 for s, p in preds.items() if truth.get(s) == p)
    n = sum(1 for s in preds if s in truth)
    if n:
        print(f"[准确率] {correct}/{n} = {correct/n:.3f}")
    else:
        print("[准确率] 没有匹配上的真值, 检查 csv 的文件名是否对应")


def main():
    if len(sys.argv) < 2:
        print("用法: python format_check.py <txt文件夹> [labels.csv]")
        return
    preds = check_format(sys.argv[1])
    if len(sys.argv) > 2 and preds:
        check_accuracy(preds, sys.argv[2])


if __name__ == "__main__":
    main()
