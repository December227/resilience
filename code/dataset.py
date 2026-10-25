# -*- coding: utf-8 -*-
"""数据加载：从 train.jsonl 随机采样数学题。"""
import json
import random
from typing import Dict, List

from answer_check import extract_answer


def load_train(path: str) -> List[Dict]:
    """读取 train.jsonl（字段: question / answer）。"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def sample_problems(path: str, sample_size: int, seed: int) -> List[Dict]:
    """随机采样 sample_size 道题, 附带标准数字答案 gold。"""
    rows = load_train(path)
    rng = random.Random(seed)
    rng.shuffle(rows)
    out = []
    for r in rows[:sample_size]:
        gold = extract_answer(r.get("answer", ""))
        if gold is None:
            continue
        out.append({
            "index": len(out),
            "question": r["question"],
            "answer": r["answer"],
            "gold": gold,
        })
    return out
