# -*- coding: utf-8 -*-
"""生成论文全部图片（图 1--8），统一字体、尺寸与样式，图内不加标题。

数据来源：
  data/resilience_metrics.json    实验一（600 题逐题结果）
  data/mechanism_results.jsonl    实验二（19 个实例的逐层效应曲线与注意力头）
  data/mechanism_summary.json     实验二（跨域相似性汇总）

用法：
  python make_paper_figures.py --font <中文字体路径> [--out ../figures]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

FONT_CANDIDATES = [
    "assets/fonts/NotoSansCJKsc-Regular.otf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]

TASKS = [
    ("boolean_expressions", "BBH-布尔表达式"),
    ("gsm_plus", "GSM-Plus"),
    ("multistep_arithmetic_two", "BBH-多步算术"),
    ("web_of_lies", "BBH-真值链"),
]

C_BLUE, C_ORANGE, C_GREEN, C_RED, C_GRAY = (
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8C8C8C")

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 200,
})


def setup_font(explicit=None):
    for path in ([explicit] if explicit else []) + FONT_CANDIDATES:
        if path and os.path.exists(path):
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[字体] {name}")
            return name
    print("[警告] 未找到中文字体，图中中文可能显示为方框")
    return None


def load_tasks(path):
    data = json.load(open(path, encoding="utf-8"))
    rows = []
    for key, label in TASKS:
        sub = [r for r in data["all"] if r["task"] == key]
        n = len(sub)
        rows.append({
            "key": key, "label": label, "n": n,
            "acc": sum(r["final_correct"] for r in sub) / n,
            "rate": sum(r["is_resilient"] for r in sub) / n,
            "seg_hist": [sum(1 for r in sub if r["error_segments"] == k) for k in (0, 1, 2)]
                        + [sum(1 for r in sub if r["error_segments"] >= 3)],
            "st_no_err": sum(1 for r in sub if r["final_correct"] and r["error_segments"] == 0),
            "st_rec": sum(1 for r in sub if r["final_correct"] and r["error_segments"] > 0),
            "st_wrong_err": sum(1 for r in sub if not r["final_correct"] and r["error_segments"] > 0),
            "st_wrong_no": sum(1 for r in sub if not r["final_correct"] and r["error_segments"] == 0),
            "n_with_err": sum(1 for r in sub if r["error_segments"] > 0),
        })
    return rows


def fig_metrics(rows, out):
    x = np.arange(len(rows))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    b1 = ax.bar(x - w / 2, [r["acc"] * 100 for r in rows], w, label="准确率", color=C_BLUE)
    b2 = ax.bar(x + w / 2, [r["rate"] * 100 for r in rows], w, label="resilience 率", color=C_ORANGE)
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            if h > 0.5:
                ax.annotate(f"{h:.1f}", (b.get_x() + b.get_width() / 2, h),
                            ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], rotation=12, ha="right")
    ax.set_ylabel("百分比 (%)")
    ax.set_ylim(0, 108)
    ax.grid(axis="y", ls=":", lw=0.6, color=C_GRAY, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_states(rows, out):
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    x = np.arange(len(rows))
    parts = [
        ("正确·无错误段", "st_no_err", C_BLUE),
        ("正确·出错后恢复", "st_rec", C_GREEN),
        ("错误·含错误段", "st_wrong_err", C_RED),
        ("错误·无错误段", "st_wrong_no", C_GRAY),
    ]
    bottom = np.zeros(len(rows))
    for label, key, color in parts:
        vals = np.array([r[key] / r["n"] * 100 for r in rows])
        ax.bar(x, vals, 0.55, bottom=bottom, label=label, color=color)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], rotation=12, ha="right")
    ax.set_ylabel("占任务内题目比例 (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", ls=":", lw=0.6, color=C_GRAY, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_segments(rows, out):
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    x = np.arange(len(rows))
    w = 0.2
    for k, (label, color) in enumerate(zip(
            ["0 段", "1 段", "2 段", "≥3 段"], [C_BLUE, C_GREEN, C_ORANGE, C_RED])):
        vals = [r["seg_hist"][k] / r["n"] * 100 for r in rows]
        ax.bar(x + (k - 1.5) * w, vals, w, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], rotation=12, ha="right")
    ax.set_ylabel("占任务内题目比例 (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", ls=":", lw=0.6, color=C_GRAY, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, title="每题的连续错误段数", title_fontsize=8,
              loc="upper center", bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def load_results(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def pair_domains(rows):
    A = [r for r in rows if r["domain"] == "math"]
    B = [r for r in rows if r["domain"] == "misleading"]
    n = min(len(A), len(B))
    return A[:n], B[:n]


def fig_effect_curves(rows, out):
    A, B = pair_domains(rows)
    ca = np.array([r["effect_curve"] for r in A])
    cb = np.array([r["effect_curve"] for r in B])
    layers = np.arange(ca.shape[1])
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for curves, color, label in ((ca, C_BLUE, "领域 A：数学推理"), (cb, C_RED, "领域 B：误导上下文")):
        m = curves.mean(0)
        ax.plot(layers, m, color=color, lw=1.8, label=f"{label}（$n={curves.shape[0]}$）")
        ax.fill_between(layers, curves.min(0), curves.max(0), color=color, alpha=0.15, lw=0)
    ax.set_xlabel("Transformer 层")
    ax.set_ylabel("效应 $E(l)$（对数概率恢复量）")
    ax.set_xlim(-0.5, layers[-1] + 1.5)
    ax.grid(ls=":", lw=0.6, color=C_GRAY, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_cka(summary, out):
    layers = sorted(summary["cka_by_layer"], key=int)
    vals = [summary["cka_by_layer"][k] for k in layers]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    bars = ax.bar([f"L{k}" for k in layers], vals, 0.55, color=C_GREEN)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9)
    ax.axhline(0.7, color=C_RED, lw=1.2, ls="--", label="阈值 0.7")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("CKA")
    ax.set_xlabel("决策点所在层")
    ax.grid(axis="y", ls=":", lw=0.6, color=C_GRAY, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_heads(rows, summary, out):
    A, B = pair_domains(rows)
    hA = set(tuple(h) for r in A for h in r["attention_heads"])
    hB = set(tuple(h) for r in B for h in r["attention_heads"])
    vals = [len(hA - hB), len(hA & hB), len(hB - hA)]
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    bars = ax.bar(["仅数学", "两域共享", "仅误导"], vals, 0.5,
                  color=[C_BLUE, C_GREEN, C_RED])
    for b, v in zip(bars, vals):
        ax.annotate(str(v), (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(vals) * 1.2)
    ax.set_ylabel("注意力头数量")
    ax.set_xlabel(f"top-10 注意力头的跨域分布（Jaccard = {summary['head_overlap_jaccard']:.3f}）")
    ax.grid(axis="y", ls=":", lw=0.6, color=C_GRAY, alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_pair_corr(rows, out):
    A, B = pair_domains(rows)
    corrs = []
    for a, b in zip(A, B):
        x = np.asarray(a["effect_curve"], float)
        y = np.asarray(b["effect_curve"], float)
        corrs.append(0.0 if (x.std() == 0 or y.std() == 0)
                     else float(np.corrcoef(x, y)[0, 1]))
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    idx = np.arange(len(corrs))
    ax.bar(idx, corrs, 0.45, color=C_BLUE)
    ax.axhline(np.mean(corrs), color=C_RED, lw=1.2, ls="--",
               label=f"均值 {np.mean(corrs):.3f}")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"配对 {i + 1}" for i in idx])
    ax.set_ylim(-1.15, 1.15)
    ax.set_ylabel("Pearson 相关")
    ax.set_xlabel("逐实例效应曲线相关（$n=4$ 对）")
    ax.grid(axis="y", ls=":", lw=0.6, color=C_GRAY, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_resilience(rows, out):
    A, B = pair_domains(rows)
    data = [[r["resilience"] for r in A], [r["resilience"] for r in B]]
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.axhspan(5, 10, color=C_GREEN, alpha=0.15, lw=0, label="目标区间 [5, 10]")
    bp = ax.boxplot(data, widths=0.4, patch_artist=True,
                    labels=[f"数学推理（$n={len(data[0])}$）",
                            f"误导上下文（$n={len(data[1])}$）"])
    for patch, color in zip(bp["boxes"], [C_BLUE, C_ORANGE]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
    ax.set_ylim(0, 11)
    ax.set_ylabel("$R_{\\mathrm{mech}}$")
    ax.grid(axis="y", ls=":", lw=0.6, color=C_GRAY, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="../data/resilience_metrics.json")
    ap.add_argument("--results", default="../data/mechanism_results.jsonl")
    ap.add_argument("--summary", default="../data/mechanism_summary.json")
    ap.add_argument("--out", default="../figures")
    ap.add_argument("--font", default=None)
    args = ap.parse_args()

    setup_font(args.font)
    os.makedirs(args.out, exist_ok=True)

    tasks = load_tasks(args.metrics)
    for r in tasks:
        print(f"{r['label']:<14} acc={r['acc']:.3f} res={r['rate']:.3f} n_err={r['n_with_err']}")
    fig_metrics(tasks, os.path.join(args.out, "fig_behavior_metrics.png"))
    fig_states(tasks, os.path.join(args.out, "fig_final_states.png"))
    fig_segments(tasks, os.path.join(args.out, "fig_error_segments.png"))

    rows = load_results(args.results)
    summary = json.load(open(args.summary, encoding="utf-8"))
    fig_effect_curves(rows, os.path.join(args.out, "fig_mech_effect_curves.png"))
    fig_cka(summary, os.path.join(args.out, "fig_mech_cka.png"))
    fig_heads(rows, summary, os.path.join(args.out, "fig_mech_head_overlap.png"))
    fig_pair_corr(rows, os.path.join(args.out, "fig_mech_per_instance_corr.png"))
    fig_resilience(rows, os.path.join(args.out, "fig_mech_resilience_by_domain.png"))
    print("[完成] 8 张图 ->", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
