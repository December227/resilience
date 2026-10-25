# -*- coding: utf-8 -*-
"""数学答案提取与正确性判定。

GSM8K 风格答案以 "#### <数字>" 结尾。对模型输出兼容多种格式。
"""
import re
from typing import Optional

from cot_utils import extract_think, split_sentences

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_number(s: str) -> Optional[str]:
    """把数字字符串规范化用于比较（去前导零、去 .0）。"""
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        f = float(s)
    except ValueError:
        return None
    if f == int(f):
        return str(int(f))
    # 统一小数位, 避免 0.5 与 0.50 不相等
    return f"{f:.10f}".rstrip("0").rstrip(".")


def extract_answer(text: str) -> Optional[str]:
    """从模型输出/标准答案中提取最终数字答案。"""
    if not text:
        return None
    # 1) 标准 "#### <num>"
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return normalize_number(m.group(1))
    # 2) <think> 之后的内容里找最后一个数字
    _, post = extract_think(text)
    if post:
        nums = NUM_RE.findall(post)
        if nums:
            return normalize_number(nums[-1])
    # 3) 全文最后一个数字
    nums = NUM_RE.findall(text)
    if nums:
        return normalize_number(nums[-1])
    return None


def check_correct(pred_text: str, gold_text: str) -> bool:
    """判定模型输出 pred_text 是否与标准答案 gold_text 数值相等。

    gold_text 既可以是纯数字("72"), 也可以是完整答案文本("...\\n#### 72")。
    """
    pred = extract_answer(pred_text)
    gold = normalize_number(gold_text)
    if gold is None:
        gold = extract_answer(gold_text)  # 可能是完整答案文本
    if pred is None or gold is None:
        return False
    return pred == gold
