# -*- coding: utf-8 -*-
"""Resilience（韧性）指标实现 —— 对应论文 Algorithm 1 与公式 (1)(2)。

论文: Thought Branches: Interpreting LLM Reasoning Requires Resampling (arXiv:2510.27484)
附录 B.1: Resilience-Based Resampling Algorithm

定义（本研究采用的清晰界定, 数学域）：
  给定思维链 CoT = [S_1, ..., S_m] 中的目标推理句 S_i。
  resilience(S_i) = 反复执行 "删除 S_i 的语义内容 -> 在其位置重采样(on-policy) ->
  检查语义内容是否在下游重新出现" 的干预次数, 直到该内容不再出现为止。
  - 若某次重采样后, 续写中没有任何句子与 S_i 的余弦相似度 > tau, 则内容被清除, 停止。
  - 若出现(模型"错误纠正/重新表述"), 则移到该重新出现的位置, 继续删除并重采样。
  resilience 值越大, 说明该推理句越"核心", 越难被移除。
"""
from typing import Dict, List, Optional

import numpy as np

from answer_check import extract_answer
from cot_utils import split_sentences


def compute_resilience(
    client,
    cot_sentences: List[str],
    i: int,
    embedder,
    tau: float,
    max_iterations: int = 6,
    n_continuations: int = 4,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens_cont: int = 384,
    mock_gold: Optional[str] = None,
    base_prefix: str = "",
) -> Dict:
    """计算第 i 个推理句的 resilience 分数（论文 Algorithm 1）。

    返回 dict:
      resilience : 干预次数（内容被清除所需的重采样轮数）
      rounds     : 每轮明细 [{round, best_sim, reappeared, best_sentence}]
      target     : 目标句文本
    """
    target = cot_sentences[i]
    prefix_before = (base_prefix + "\n" if base_prefix else "") + "\n".join(cot_sentences[:i])  # 删除 S_i 后的前缀

    k = 0
    rounds = []
    current_prefix = prefix_before

    while k < max_iterations:
        # 1) 在当前前缀位置重采样（on-policy），生成若干续写样本
        conts = client.generate(
            current_prefix,
            n=n_continuations,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens_cont,
            mock_target=target,
            mock_round=k,
            mock_gold=mock_gold,
        )
        # 2) 在续写中寻找与目标句最相似的句子
        best_sim = -1.0
        best_sentence = None
        best_cont_sents = None
        best_j = -1
        for cont in conts:
            cont_sents = split_sentences(cont)
            if not cont_sents:
                continue
            sims = embedder.similarity(target, cont_sents)
            j = int(np.argmax(sims))
            if float(sims[j]) > best_sim:
                best_sim = float(sims[j])
                best_sentence = cont_sents[j]
                best_cont_sents = cont_sents
                best_j = j

        reappeared = best_sim > tau
        rounds.append({
            "round": k,
            "best_sim": round(best_sim, 4),
            "reappeared": reappeared,
            "best_sentence": best_sentence,
        })

        if reappeared:
            # 3) 内容在下游重新出现 -> 干预次数 +1, 移到重新出现的位置继续删除
            k += 1
            if best_cont_sents is not None and best_j > 0:
                current_prefix = current_prefix + "\n" + "\n".join(best_cont_sents[:best_j])
            # 若 best_j == 0: 内容紧接在重采样起点出现, 前缀不变, 下一轮继续重采样
        else:
            # 4) 内容被清除
            break

    return {
        "target": target,
        "resilience": k,
        "rounds": rounds,
        "tau": tau,
    }


def compute_counterfactual_importance(
    client,
    cot_sentences: List[str],
    i: int,
    embedder,
    tau: float,
    n_rollouts: int = 20,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 512,
    mock_gold: Optional[str] = None,
) -> float:
    """反事实重要性（论文公式 1）的数学域适配。

    importance(S_i) = D_KL[ p(A|S_i 被清除且未重现) || p(A|S_i 存在) ]
    其中 A 为最终答案(取数字答案的分布)。带 Laplace 平滑。
    """
    target = cot_sentences[i]
    prefix_before = "\n".join(cot_sentences[:i])
    prefix_with = "\n".join(cot_sentences[: i + 1])

    # p(A | S_i 存在)：从包含 S_i 的前缀重采样
    conts_with = client.generate(
        prefix_with, n=n_rollouts, temperature=temperature, top_p=top_p,
        max_tokens=max_tokens, mock_target=None, mock_gold=mock_gold,
    )
    dist_with = _answer_distribution(conts_with)

    # p(A | S_i 被清除且未在下游重现)：从删除 S_i 的前缀重采样, 过滤掉重现 rollouts
    conts_without = client.generate(
        prefix_before, n=n_rollouts, temperature=temperature, top_p=top_p,
        max_tokens=max_tokens, mock_target=target, mock_round=999, mock_gold=mock_gold,
    )
    kept = []
    for c in conts_without:
        sents = split_sentences(c)
        if sents and embedder.max_similarity(target, sents[:3]) > tau:
            continue  # 目标内容重现, 不符合"完全清除"条件
        kept.append(c)
    dist_without = _answer_distribution(kept)

    return _kl_divergence(dist_without, dist_with)


def _answer_distribution(texts: List[str]) -> Dict[str, int]:
    counts = {}
    for t in texts:
        a = extract_answer(t)
        a = a if a is not None else "<none>"
        counts[a] = counts.get(a, 0) + 1
    return counts


def _kl_divergence(p: Dict[str, int], q: Dict[str, int]) -> float:
    """带 Laplace 平滑的 KL(P||Q)。"""
    keys = set(p) | set(q)
    alpha = 1.0
    total_p = sum(p.values()) + alpha * len(keys)
    total_q = sum(q.values()) + alpha * len(keys)
    kl = 0.0
    for key in keys:
        pp = (p.get(key, 0) + alpha) / total_p
        qq = (q.get(key, 0) + alpha) / total_q
        if pp > 0:
            kl += pp * np.log(pp / qq)
    return float(kl)
