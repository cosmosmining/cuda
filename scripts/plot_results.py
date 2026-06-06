#!/usr/bin/env python3
"""Render the results figures from data/results.csv.

Produces, under docs/results/:
    gemm_gflops.png       fp32 GEMM: ours vs cuBLAS vs PyTorch eager
    int8_gops.png         INT8 matmul: ours (__dp4a) vs cuBLAS (tensor-core)
    attention_tokens.png  fused attention decode-step throughput vs PyTorch SDPA

If the CSV still carries the synthetic-data banner, every figure is stamped
"ILLUSTRATIVE" so a placeholder plot can never be mistaken for a measurement.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _csvio import load_rows, is_illustrative  # noqa: E402

OUT = os.path.join("docs", "results")


def stamp(fig, illustrative: bool):
    if illustrative:
        fig.text(0.5, 0.5, "ILLUSTRATIVE", fontsize=44, color="gray",
                 alpha=0.18, ha="center", va="center", rotation=25,
                 zorder=10)
        fig.text(0.99, 0.01, "synthetic placeholder data — regenerate with "
                 "bench/run_all.sh", fontsize=7, color="gray", ha="right",
                 va="bottom")


def by_kernel(rows, name):
    return [r for r in rows if r["kernel"] == name]


def grouped_bar(ax, labels, series, ylabel, title):
    n = len(series)
    width = 0.8 / max(n, 1)
    x = range(len(labels))
    for i, (sname, vals) in enumerate(series):
        ax.bar([xi + i * width for xi in x], vals, width, label=sname)
    ax.set_xticks([xi + width * (n - 1) / 2 for xi in x])
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)


def plot_gemm(rows, illustrative):
    ours = by_kernel(rows, "gemm_tiled_fp32")
    cub = {r["shape"]: r["cublas_gflops"] for r in ours}
    pyt = {r["shape"]: r["ours_gflops"]
           for r in by_kernel(rows, "gemm_pytorch_eager")}
    labels = [r["shape"] for r in ours]
    series = [
        ("tiny-infer (ours)", [r["ours_gflops"] / 1000 for r in ours]),
        ("cuBLAS", [cub[s] / 1000 for s in labels]),
        ("PyTorch eager", [pyt.get(s, 0) / 1000 for s in labels]),
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    grouped_bar(ax, labels, series, "TFLOP/s",
                "FP32 GEMM throughput (higher is better)")
    stamp(fig, illustrative)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "gemm_gflops.png"), dpi=130)
    plt.close(fig)


def plot_int8(rows, illustrative):
    ours = by_kernel(rows, "matmul_int8")
    labels = [r["shape"] for r in ours]
    series = [
        ("tiny-infer (__dp4a)", [r["ours_gflops"] / 1000 for r in ours]),
        ("cuBLAS (tensor-core)", [r["cublas_gflops"] / 1000 for r in ours]),
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    grouped_bar(ax, labels, series, "TOP/s",
                "INT8 matmul throughput (higher is better)")
    stamp(fig, illustrative)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "int8_gops.png"), dpi=130)
    plt.close(fig)


def plot_attention(rows, illustrative):
    ours = by_kernel(rows, "attn_fused_fp32")
    pyt = {r["shape"]: r["tokens_per_s"]
           for r in by_kernel(rows, "attn_pytorch_sdpa")}
    labels = [r["shape"] for r in ours]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(labels, [r["tokens_per_s"] / 1000 for r in ours], "o-",
            label="tiny-infer fused")
    if any(s in pyt for s in labels):
        ax.plot(labels, [pyt.get(s, 0) / 1000 for s in labels], "s--",
                label="PyTorch SDPA (FlashAttention-2)")
    ax.set_ylabel("decode steps/sec (thousands)")
    ax.set_title("Fused attention decode throughput vs context length")
    ax.legend()
    ax.grid(alpha=0.3)
    stamp(fig, illustrative)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "attention_tokens.png"), dpi=130)
    plt.close(fig)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/results.csv"
    os.makedirs(OUT, exist_ok=True)
    rows = load_rows(csv_path)
    illustrative = is_illustrative(csv_path)
    plot_gemm(rows, illustrative)
    plot_int8(rows, illustrative)
    plot_attention(rows, illustrative)
    tag = " (ILLUSTRATIVE)" if illustrative else ""
    print(f"[plot_results] wrote 3 figures to {OUT}/{tag}")


if __name__ == "__main__":
    main()
