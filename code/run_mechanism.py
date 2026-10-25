#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨领域 resilience 表征相似性实验的主流程。

流程分四步：
  1) 从 GSM8K 采样题目
  2) 构造两个领域的实例，数学推理与误导上下文各一组，韧性校准到目标区间
  3) 对每个高韧性实例做机制定位，包括激活补丁的逐层效应曲线、注意力头定位、决策点残差流
  4) 比较两个领域的效应曲线相关性、决策点 CKA 与 top 头重叠

用法：

    python run_mechanism.py --config config.yaml
    python run_mechanism.py --config config.yaml --model models/DeepSeek-R1-Distill-Qwen-1.5B
    python run_mechanism.py --config config.yaml --sample-size 20 --limit 5

先把模型换成 1.5B 跑通全流程，再换 14B，能省不少排查时间。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None, help="覆盖模型路径(小模型验证用)")
    ap.add_argument("--sample-size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 道题")
    ap.add_argument("--no-mechanism", action="store_true", help="只做行为 resilience, 跳过机制")
    args = ap.parse_args()

    from utils import ensure_hf_env, load_config, seed_all
    ensure_hf_env()
    cfg = load_config(_abs(args.config))
    seed_all(cfg["output"]["seed"])

    model_cfg = cfg["model"]
    if args.model:
        model_cfg["name"] = args.model
    model_cfg["name"] = _abs(model_cfg["name"])
    model_cfg["_sampling"] = cfg["sampling"]
    res_cfg = cfg["resilience"]
    res_cfg["embedding_model"] = _abs(res_cfg["embedding_model"])
    exp_cfg = cfg["experiment"]
    mech_cfg = cfg["mechanism"]
    out_cfg = cfg["output"]
    for k in ("dir", "figures_dir", "report_dir", "cache_dir", "results_file", "summary_file"):
        out_cfg[k] = _abs(out_cfg[k])
    for d in ("cache_dir", "figures_dir", "report_dir"):
        os.makedirs(out_cfg[d], exist_ok=True)

    from dataset import sample_problems
    from embeddings import Embedder
    from llm_client import LLMClient  # 只有 transformers 后端才会真的加载模型
    from domains import build_instances

    problems = sample_problems(_abs(exp_cfg["data_source"]), args.sample_size or exp_cfg["sample_size"],
                               exp_cfg["sample_seed"])
    if args.limit:
        problems = problems[: args.limit]
    print(f"[数据] {len(problems)} 道题 (来源: {exp_cfg['data_source']})")

    client = LLMClient(model_cfg)
    embedder = Embedder(res_cfg["embedding_model"], device=res_cfg.get("embedding_device", "cpu"))

    # ---------- 行为 resilience: 构造实例(目标区间 5-10) ----------
    tmin, tmax = exp_cfg["target_min"], exp_cfg["target_max"]
    instances = []
    for domain in ("math", "misleading"):
        inst = build_instances(
            problems, domain, client, embedder, cfg,
            n_base_cots=exp_cfg.get("n_base_cots", cfg["sampling"]["n_base_cots"]),
            target_min=tmin, target_max=tmax,
            max_instances=exp_cfg["max_instances"],
            misleading_level=exp_cfg.get("misleading_level", 3),
        )
        print(f"[领域 {domain}] 高韧性实例(区间[{tmin},{tmax}]): {len(inst)} 个")
        instances.extend(inst)

    # ---------- 机制定位 + 跨域相似性 ----------
    if not args.no_mechanism and instances:
        import mechanism as mech
        model = client.backend.model
        tokenizer = client.backend.tokenizer
        model.eval()
        device = next(model.parameters()).device

        def tokenize_parts(parts, max_tokens):
            """分段 tokenize 并拼接, 返回 (ids, target_span)。parts: [(text, is_target), ...]"""
            all_ids = []
            t_start = t_end = None
            for text, is_target in parts:
                ids = tokenizer.encode(text, add_special_tokens=False)
                if is_target:
                    t_start = len(all_ids)
                    t_end = len(all_ids) + len(ids)
                all_ids.extend(ids)
            if len(all_ids) > max_tokens:
                # 截断前缀部分, 保留 target 与结尾
                keep = max_tokens
                if t_end is not None and t_end > keep:
                    # 保证 target 完整保留
                    over = t_end - keep
                    all_ids = all_ids[over:]
                    t_start -= over
                    t_end -= over
                elif len(all_ids) > keep:
                    # target 在前, 截掉后面的后续句
                    all_ids = all_ids[:keep]
            return all_ids, (t_start, t_end)

        results = []
        for inst in tqdm(instances, desc="机制定位"):
            cot = inst["cot_sentences"]
            ti = inst["target_idx"]
            base = inst["base_prompt"]
            # R1 蒸馏模型需要特殊 token 触发 <think> 格式（与 backend._build_prompt 一致）
            R1_PROMPT_PREFIX = "<｜begin▁of▁sentence｜><｜User｜>"
            R1_PROMPT_SUFFIX = "<｜Assistant｜>"
            base_fmt = f"{R1_PROMPT_PREFIX}{base}{R1_PROMPT_SUFFIX}"
            # 1) 补丁序列: 前缀 + 目标句(截断)
            parts_patch = [(base_fmt, False)]
            for s in cot[:ti]:
                parts_patch.append((" " + s, False))
            parts_patch.append((" " + cot[ti], True))
            ids, span = tokenize_parts(parts_patch, mech_cfg["patch_max_tokens"])
            if span[0] is None or span[1] - span[0] < 1:
                continue
            input_ids = torch.tensor([ids], device=device)

            # 2) clean / corrupt
            clean_states, clean_logits = mech.forward_with_states(model, input_ids)
            ref_token = int(clean_logits[0, -1].argmax())
            corrupt_states, corrupt_logits = mech.forward_with_corrupted_embedding(
                model, input_ids, span)
            last = input_ids.shape[1] - 1
            clean_outcome = mech._logp(clean_logits, last, ref_token)
            corrupt_outcome = mech._logp(corrupt_logits, last, ref_token)
            effects = mech.causal_trace(
                model, input_ids, span[1] - 1, clean_states, corrupt_outcome, ref_token)

            # 3) 注意力(完整 CoT 到目标后几句)
            attn_results = []
            parts_attn = [(base_fmt, False)]
            up_to = min(len(cot), ti + 4)
            for j, s in enumerate(cot[:up_to]):
                parts_attn.append((" " + s, j == ti))
            ids_a, span_a = tokenize_parts(parts_attn, mech_cfg["patch_max_tokens"])
            if span_a[0] is not None and span_a[1] - span_a[0] >= 1:
                attn_ids = torch.tensor([ids_a], device=device)
                attentions = mech.attention_weights(model, attn_ids)
                heads = mech.top_attention_heads(attentions, span_a[0], span_a[1], k=mech_cfg["top_heads_k"])
                attn_results = [(l, h) for _, l, h in heads]

            # 4) 决策点残差流(选层)用于 CKA
            rep = {l: clean_states[l][0, span[1] - 1].float().cpu().numpy()
                   for l in mech_cfg["cka_layers"]}

            results.append({
                "problem_index": inst["problem_index"],
                "domain": inst["domain"],
                "target_sentence": inst["target_sentence"],
                "resilience": inst["resilience"],
                "effect_curve": effects.tolist(),
                "clean_outcome": clean_outcome,
                "corrupt_outcome": corrupt_outcome,
                "attention_heads": attn_results,
                "representation": {str(l): rep[l].tolist() for l in rep},
                "n_layers": len(effects),
            })

        with open(out_cfg["results_file"], "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # ---------- 跨域相似性 ----------
        def domain_rows(dom):
            return [r for r in results if r["domain"] == dom]

        A, B = domain_rows("math"), domain_rows("misleading")
        summary = {}
        if A and B:
            n = min(len(A), len(B))
            A, B = A[:n], B[:n]
            curvesA = np.array([r["effect_curve"] for r in A])   # [n, L]
            curvesB = np.array([r["effect_curve"] for r in B])
            meanA, meanB = curvesA.mean(0), curvesB.mean(0)
            corr_mean = mech.pearson(meanA, meanB)
            per_inst_corr = [mech.pearson(a, b) for a, b in zip(curvesA, curvesB)]
            # CKA per layer
            cka_by_layer = {}
            for l in mech_cfg["cka_layers"]:
                X = np.array([r["representation"][str(l)] for r in A])
                Y = np.array([r["representation"][str(l)] for r in B])
                cka_by_layer[str(l)] = mech.cka(X, Y)
            # attention head overlap (top-k sets per domain)
            headsA = set(h for r in A for h in r["attention_heads"])
            headsB = set(h for r in B for h in r["attention_heads"])
            inter = len(headsA & headsB)
            union = len(headsA | headsB) or 1
            head_overlap = inter / union
            summary = {
                "n_math": len(A), "n_misleading": len(B),
                "resilience_math_mean": float(np.mean([r["resilience"] for r in A])),
                "resilience_misleading_mean": float(np.mean([r["resilience"] for r in B])),
                "pearson_mean_curves": corr_mean,
                "pearson_per_instance_mean": float(np.mean(per_inst_corr)),
                "cka_by_layer": cka_by_layer,
                "head_overlap_jaccard": head_overlap,
                "n_heads_math": len(headsA), "n_heads_misleading": len(headsB),
                "decision": "表征相似(支持共享机制)" if (abs(corr_mean) > mech_cfg["similarity_significance"] or
                                                          max(cka_by_layer.values()) > 0.7) else "表征不相似(领域特异)",
                "effect_curve_math": meanA.tolist(),
                "effect_curve_misleading": meanB.tolist(),
            }
        else:
            summary = {"n_math": len(A), "n_misleading": len(B),
                       "decision": "实例不足, 无法比较(请调整校准参数提高命中数)"}

        with open(out_cfg["summary_file"], "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("=== 跨域相似性 ===")
        print("  效应曲线 Pearson(均值):", summary.get("pearson_mean_curves"))
        print("  CKA(逐层):", summary.get("cka_by_layer"))
        print("  注意力头重叠(Jaccard):", summary.get("head_overlap_jaccard"))
        print("  结论:", summary.get("decision"))
    else:
        # 仅行为模式
        summary = {"n_math": len([i for i in instances if i["domain"] == "math"]),
                   "n_misleading": len([i for i in instances if i["domain"] == "misleading"]),
                   "decision": "仅行为模式(未跑机制)"}
        with open(out_cfg["summary_file"], "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[完成] results -> {out_cfg['results_file']}\n       summary -> {out_cfg['summary_file']}")


if __name__ == "__main__":
    main()
