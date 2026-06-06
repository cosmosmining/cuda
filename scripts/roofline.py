#!/usr/bin/env python3
"""Roofline plot for the tiny-infer kernels on an H100 SXM.

Draws the HBM memory roof and the fp32 / bf16-TC / int8-TC compute ceilings,
then places each kernel's achieved throughput at its arithmetic intensity
(computed from the largest swept shape in data/results.csv).

The hardware peaks below are *vendor reference figures* for an H100 SXM5 and are
themselves illustrative — confirm against your exact SKU and clocks. The figure
is stamped ILLUSTRATIVE whenever the CSV still holds synthetic data.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _csvio import load_rows, is_illustrative  # noqa: E402

# --- H100 SXM5 vendor-reference peaks (illustrative; verify for your SKU). ---
HBM_BW_TBS = 3.35            # TB/s  -> bytes/s below
PEAK_FP32_TFLOPS = 66.9      # non-tensor-core fp32
PEAK_BF16_TC_TFLOPS = 989.0  # dense bf16/fp16 tensor core
PEAK_INT8_TC_TOPS = 1979.0   # dense int8 tensor core

HBM_BW = HBM_BW_TBS * 1e12
CEILINGS = [
    ("fp32 (CUDA cores)", PEAK_FP32_TFLOPS * 1e12, "C0"),
    ("bf16 tensor core", PEAK_BF16_TC_TFLOPS * 1e12, "C1"),
    ("int8 tensor core", PEAK_INT8_TC_TOPS * 1e12, "C2"),
]


def parse_n(shape: str) -> int:
    # "8192^3" -> 8192 ; "seq4096_d64" -> 4096
    if "^" in shape:
        return int(shape.split("^")[0])
    return int(shape.split("seq")[1].split("_")[0])


def achieved_points(rows):
    """(label, arithmetic_intensity[flop/byte], achieved[op/s], color) per kernel.

    AI uses each kernel's dominant global-memory traffic model:
      GEMM  NxNxN fp32 : 2N^3 flop / (3 N^2 * 4 B)          = N/6
      attn  fused fp32 : 4 S^2 d / (4 S d * 4 B)            = S/4   (no S matrix)
      int8  NxNxN      : 2N^3 op  / (2 N^2 * 1 B + N^2 * 4) = N/3
    """
    def last(kernel):
        rs = [r for r in rows if r["kernel"] == kernel]
        return max(rs, key=lambda r: parse_n(r["shape"])) if rs else None

    pts = []
    g = last("gemm_tiled_fp32")
    if g:
        n = parse_n(g["shape"])
        pts.append(("GEMM fp32", n / 6.0, g["ours_gflops"] * 1e9, "C0"))
    a = last("attn_fused_fp32")
    if a:
        s = parse_n(a["shape"])
        pts.append(("attn fp32", s / 4.0, a["ours_gflops"] * 1e9, "C3"))
    q = last("matmul_int8")
    if q:
        n = parse_n(q["shape"])
        pts.append(("matmul int8", n / 3.0, q["ours_gflops"] * 1e9, "C2"))
    return pts


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/results.csv"
    rows = load_rows(csv_path)
    illustrative = is_illustrative(csv_path)

    ai = np.logspace(-1, 4, 400)  # flop/byte
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Memory roof (diagonal) and compute ceilings (horizontal).
    ax.loglog(ai, HBM_BW * ai, "k-", lw=1.5,
              label=f"HBM roof {HBM_BW_TBS:.2f} TB/s")
    for name, peak, color in CEILINGS:
        ax.axhline(peak, color=color, ls="--", lw=1.2, alpha=0.8,
                   label=f"{name} {peak/1e12:.0f} T")

    # Achieved points.
    for label, x, y, color in achieved_points(rows):
        ax.plot(x, y, "o", color=color, ms=9, mec="black", zorder=5)
        ax.annotate(f"{label}\n{y/1e12:.0f} T @ AI={x:.0f}", (x, y),
                    textcoords="offset points", xytext=(8, -4), fontsize=8)

    ax.set_xlabel("arithmetic intensity (FLOP or OP / byte)")
    ax.set_ylabel("attained throughput (FLOP/s or OP/s)")
    ax.set_title("tiny-infer roofline — H100 SXM (vendor-peak reference)")
    ax.set_ylim(1e11, 3e15)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)

    if illustrative:
        fig.text(0.5, 0.5, "ILLUSTRATIVE", fontsize=46, color="gray",
                 alpha=0.18, ha="center", va="center", rotation=22)
        fig.text(0.99, 0.005, "synthetic placeholder data + vendor-peak "
                 "reference — regenerate with bench/run_all.sh", fontsize=7,
                 color="gray", ha="right", va="bottom")

    os.makedirs("docs", exist_ok=True)
    fig.tight_layout()
    fig.savefig("docs/roofline.png", dpi=130)
    plt.close(fig)
    tag = " (ILLUSTRATIVE)" if illustrative else ""
    print(f"[roofline] wrote docs/roofline.png{tag}")


if __name__ == "__main__":
    main()
