#!/usr/bin/env python3
"""Generate the two most impactful charts from eval results.

Chart 1: Eval scores across all harnesses — shows where the 4B model
         excels vs struggles on agentic build tasks.

Chart 2: Memory vs decode throughput — baseline vs eval harness overlay,
         showing the model's efficiency envelope and that the harness
         adds no overhead.
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_data():
    base_dir = Path(__file__).parent / "results"
    with open(base_dir / "full_eval_report.json") as f:
        evl = json.load(f)
    with open(base_dir / "baseline_benchmark.json") as f:
        base = json.load(f)
    return evl, base


def chart1_eval_scores(evl, out_path):
    """Grouped bar chart: all harness dimensions side-by-side."""

    fig, ax = plt.subplots(figsize=(14, 6))

    harnesses = []
    dimensions = {}

    # Planning tiers
    for tier in ("short", "medium", "long"):
        s = evl["planning"][tier]["score"]
        label = f"Planning\n({tier})"
        harnesses.append(label)
        for dim in ("step_completeness", "dependency_ordering", "technical_accuracy", "bonus_coverage"):
            dimensions.setdefault(dim, []).append(s[dim])

    # Error diagnosis
    s = evl["error_diagnosis"]["score"]
    harnesses.append("Error\nDiagnosis")
    dimensions.setdefault("step_completeness", []).append(s["root_cause_identification"])
    dimensions.setdefault("dependency_ordering", []).append(s["fix_quality"])
    dimensions.setdefault("technical_accuracy", []).append(s["evidence_citation"])
    dimensions.setdefault("bonus_coverage", []).append(0)

    # Tool use
    ts = evl["tool_use"]["overall_score"]
    harnesses.append("Tool\nUse")
    dimensions.setdefault("step_completeness", []).append(ts["action_correctness"])
    dimensions.setdefault("dependency_ordering", []).append(ts["reasoning_quality"])
    dimensions.setdefault("technical_accuracy", []).append(ts["context_retention"])
    dimensions.setdefault("bonus_coverage", []).append(ts["efficiency"])

    dim_labels = {
        "step_completeness": "Completeness / Root Cause / Actions",
        "dependency_ordering": "Ordering / Fix Quality / Reasoning",
        "technical_accuracy": "Accuracy / Evidence / Context",
        "bonus_coverage": "Bonus / Efficiency",
    }
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    x = np.arange(len(harnesses))
    n_dims = len(dimensions)
    width = 0.18
    offsets = np.linspace(-(n_dims - 1) * width / 2, (n_dims - 1) * width / 2, n_dims)

    for i, (dim_key, values) in enumerate(dimensions.items()):
        bars = ax.bar(x + offsets[i], values, width, label=dim_labels[dim_key],
                      color=colors[i], alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    # Overall score line
    overalls = []
    for tier in ("short", "medium", "long"):
        overalls.append(evl["planning"][tier]["score"]["overall_score"])
    overalls.append(evl["error_diagnosis"]["score"]["overall_score"])
    overalls.append(evl["tool_use"]["overall_score"]["overall_score"])

    ax.plot(x, overalls, "k--o", linewidth=2, markersize=8, label="Overall Score", zorder=5)
    for xi, ov in zip(x, overalls):
        ax.text(xi, ov + 0.03, f"{ov:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#333")

    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Gemma 4 E4B — Agentic Eval Scores Across All Harnesses",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(harnesses, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1.0, color="#ccc", linestyle=":", linewidth=0.8)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Chart 1 saved: {out_path}")


def chart2_memory_vs_throughput(evl, base, out_path):
    """Scatter plot: Peak RSS vs decode tok/s for baseline and eval runs."""

    fig, ax = plt.subplots(figsize=(12, 7))

    # Baseline points
    base_rss = [r["peak_rss_mb"] for r in base["runs"]]
    base_decode = [r["decode_tok_per_sec"] for r in base["runs"]]
    base_labels = [r["prompt_label"] for r in base["runs"]]
    base_prefill = [r["prefill_tok_per_sec"] for r in base["runs"]]

    # Eval points
    eval_rss = []
    eval_decode = []
    eval_labels = []
    eval_prefill = []
    eval_scores = []

    for tier in ("short", "medium", "long"):
        m = evl["planning"][tier]["inference"]["metrics"]
        eval_rss.append(m["peak_rss_mb"])
        eval_decode.append(m["decode_tokens_per_sec"])
        eval_prefill.append(m.get("prefill_tokens_per_sec", 0))
        eval_labels.append(f"Plan ({tier})")
        eval_scores.append(evl["planning"][tier]["score"]["overall_score"])

    m = evl["error_diagnosis"]["inference"]["metrics"]
    eval_rss.append(m["peak_rss_mb"])
    eval_decode.append(m["decode_tokens_per_sec"])
    eval_prefill.append(m.get("prefill_tokens_per_sec", 0))
    eval_labels.append("Err Diag")
    eval_scores.append(evl["error_diagnosis"]["score"]["overall_score"])

    for t in evl["tool_use"]["turns"]:
        m = t["inference"]["metrics"]
        eval_rss.append(m["peak_rss_mb"])
        eval_decode.append(m["decode_tokens_per_sec"])
        eval_prefill.append(m.get("prefill_tokens_per_sec", 0))
        eval_labels.append(f"Tool T{t['turn']}")
        eval_scores.append(t["scores"]["action_score"])

    # Plot baseline
    ax.scatter(base_rss, base_decode, s=200, c="#90CAF9", edgecolors="#1565C0",
               linewidth=2, zorder=4, label="Baseline (raw)", marker="s")
    for x, y, lbl in zip(base_rss, base_decode, base_labels):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(8, 8),
                    fontsize=8, color="#1565C0", fontweight="bold")

    # Plot eval — color by score
    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(0.4, 1.0)
    scatter = ax.scatter(eval_rss, eval_decode, s=250, c=eval_scores, cmap=cmap,
                         norm=norm, edgecolors="#333", linewidth=2, zorder=5,
                         label="Eval harness", marker="o")
    for x, y, lbl in zip(eval_rss, eval_decode, eval_labels):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(8, -12),
                    fontsize=8, color="#333", fontweight="bold")

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Eval Score", fontsize=10, fontweight="bold")

    # Draw the "stable decode band"
    ax.axhspan(12.4, 14.5, alpha=0.08, color="#4CAF50", zorder=1)
    ax.text(2750, 14.6, "Stable decode band (12.5–14.3 tok/s)",
            fontsize=9, color="#388E3C", fontstyle="italic")

    # Annotations
    ax.annotate("KV-cache growth →", xy=(3100, 11.8), fontsize=9, color="#666",
                fontstyle="italic",
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.5),
                xytext=(2850, 11.5))

    ax.set_xlabel("Peak RSS (MB)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Decode Speed (tok/s)", fontsize=12, fontweight="bold")
    ax.set_title("Gemma 4 E4B — Memory vs Throughput: Baseline vs Eval Harness",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(2600, 3700)
    ax.set_ylim(11.0, 15.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Chart 2 saved: {out_path}")


def main():
    evl, base = load_data()
    out_dir = Path(__file__).parent / "results"

    chart1_eval_scores(evl, out_dir / "chart_eval_scores.png")
    chart2_memory_vs_throughput(evl, base, out_dir / "chart_memory_vs_throughput.png")

    # Copy to artifacts
    import shutil
    artifacts = Path("/opt/cursor/artifacts")
    artifacts.mkdir(exist_ok=True)
    shutil.copy(out_dir / "chart_eval_scores.png", artifacts / "chart_eval_scores.png")
    shutil.copy(out_dir / "chart_memory_vs_throughput.png", artifacts / "chart_memory_vs_throughput.png")
    print("Charts copied to /opt/cursor/artifacts/")


if __name__ == "__main__":
    main()
