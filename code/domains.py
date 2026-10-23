# -*- coding: utf-8 -*-
"""双领域构造：数学推理(math) 与 误导上下文(misleading)。

统一流程:
  1) 生成 base CoT(可含误导前缀)
  2) 保留答对的 CoT
  3) 对每条 CoT 的每个句子计算 Algorithm-1 resilience
  4) 选出 resilience 落在目标区间(如 5-10)的句子作为"目标句"实例

领域差异仅体现在"前缀是否含误导上下文"与"目标句的语义类型"上,
机制协议(激活补丁)对两者完全一致。
"""
from typing import Dict, List, Optional

from .answer_check import check_correct
from .cot_utils import get_reasoning_sentences
from .resilience import compute_resilience


def make_misleading_prompt(question: str, gold: str, level: int = 3, max_level: int = 10) -> str:
    """构造误导提示。level 越大误导越强(gold 改成错误数)。"""
    wrong = str((int(float(gold)) + 1) if gold.replace('-', '').isdigit() else "the other number")
    phrases = [
        "Some people think the answer might be {X}.",
        "A preliminary analysis suggested the answer could be {X}.",
        "Several solvers reported the answer as {X}.",
        "Note that previous analyses converged on {X}.",
        "It is worth noting many experts believe the answer is {X}.",
        "Recent results strongly suggest the answer is {X}.",
        "There is growing consensus that the answer is {X}.",
        "Most sources confirm the answer is {X}.",
        "The answer is widely accepted to be {X}.",
        "It is now established that the answer is {X}.",
    ]
    idx = min(level, max_level) - 1
    hint = phrases[idx].format(X=wrong)
    return f"{hint}\n\n{question}"


def _select_resilient_sentences(
    cot_text: str,
    gold: str,
    client,
    embedder,
    tau: float,
    k_max: int,
    n_cont: int,
    temp: float,
    top_p: float,
    max_cont_tokens: int,
    target_min: int,
    target_max: int,
    base_prefix: str = "",
    require_correct: bool = True,
) -> List[Dict]:
    """对一条 CoT 逐句算 resilience, 返回命中目标区间的句子实例。"""
    if require_correct and not check_correct(cot_text, gold):
        return []
    sents = get_reasoning_sentences(cot_text)
    if len(sents) < 2:
        return []
    out = []
    for i, s in enumerate(sents):
        r = compute_resilience(
            client, sents, i, embedder, tau=tau,
            max_iterations=k_max, n_continuations=n_cont,
            temperature=temp, top_p=top_p, max_tokens_cont=max_cont_tokens,
            base_prefix=base_prefix,
        )
        if target_min <= r["resilience"] <= target_max:
            out.append({
                "target_idx": i,
                "target_sentence": s,
                "resilience": r["resilience"],
                "cot_sentences": sents,
                "rounds": r["rounds"],
            })
    return out


def build_instances(
    problems: List[Dict],
    domain: str,
    client,
    embedder,
    cfg,
    n_base_cots: int,
    target_min: int,
    target_max: int,
    max_instances: int = 30,
    misleading_level: int = 3,
) -> List[Dict]:
    """构造某领域的实例列表。

    返回每条: {problem_index, question, gold, domain, base_prompt,
               target_idx, target_sentence, resilience, cot_sentences}
    """
    sampling = cfg["sampling"]
    res = cfg["resilience"]
    instances = []
    for pi, prob in enumerate(problems):
        question = prob["question"]
        gold = prob["gold"]
        base_prompt = question if domain == "math" else make_misleading_prompt(question, gold, misleading_level)
        base_prefix = base_prompt  # 重采样前缀
        require_correct = (domain == "math")  # misleading 领域不要求答案正确
        print(f"  [{domain}] 问题 {pi+1}/{len(problems)}: 生成 {n_base_cots} 条 CoT...", flush=True)
        cots = client.generate(base_prompt, n=n_base_cots)
        for ci, cot in enumerate(cots):
            print(f"    CoT {ci+1}: 计算 resilience...", flush=True)
            hits = _select_resilient_sentences(
                cot, gold, client, embedder,
                tau=res["similarity_threshold"],
                k_max=res["max_iterations"],
                n_cont=sampling["n_continuations"],
                temp=sampling["temperature"], top_p=sampling["top_p"],
                max_cont_tokens=sampling["max_new_tokens_cont"],
                target_min=target_min, target_max=target_max,
                base_prefix=base_prefix,
                require_correct=require_correct,
            )
            for h in hits:
                instances.append({
                    "problem_index": prob["index"],
                    "question": question,
                    "gold": gold,
                    "domain": domain,
                    "base_prompt": base_prompt,
                    **h,
                })
                if len(instances) >= max_instances:
                    return instances
    return instances
