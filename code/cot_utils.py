# -*- coding: utf-8 -*-
"""思维链(CoT)解析工具：<think> 提取、句子切分。"""
import re
from typing import List, Tuple

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")


def extract_think(text: str) -> Tuple[str, str]:
    """返回 (think 内容, think 之后的内容)。若没有 <think> 标签, 整个文本视为思考内容。"""
    m = THINK_RE.search(text)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    return text.strip(), ""


def split_sentences(text: str) -> List[str]:
    """按句号/问号/叹号切分句子（避免切坏小数 48.24）。"""
    if not text:
        return []
    text = text.strip().replace("\n", " ")
    # 保护 "####" 答案标记：临时替换再还原
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # 去掉纯标记
        if re.fullmatch(r"#{2,}\s*\d*", p):
            continue
        out.append(p)
    return out


def get_reasoning_sentences(cot_text: str) -> List[str]:
    """从完整输出中提取用于分析的推理句列表（<think> 内的句子）。"""
    think, post = extract_think(cot_text)
    sents = split_sentences(think)
    return sents


def get_reasoning_text(cot_text: str) -> str:
    think, _ = extract_think(cot_text)
    return think
