#!/usr/bin/env python3
"""PyTorch eager baselines (matmul, scaled-dot-product attention).

Appends rows to the same results CSV the CUDA benchmarks write, so the plot
scripts can place "PyTorch eager" next to the hand-written kernels and cuBLAS.

Usage:
    python bench/bench_pytorch.py [csv_path]

Requires a CUDA-enabled PyTorch build. On a CPU-only box it prints a notice and
exits 0 (so CI does not fail); no numbers are fabricated.
"""
import csv
import os
import sys
import time


def main() -> int:
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/results.csv"
    try:
        import torch
    except ImportError:
        print("[bench_pytorch] torch not installed; skipping (install a CUDA "
              "build to populate PyTorch-eager rows).")
        return 0
    if not torch.cuda.is_available():
        print("[bench_pytorch] CUDA not available to PyTorch; skipping.")
        return 0

    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False  # match fp32 kernels
    rows = []

    def timed(fn, iters=50, warmup=10):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3  # ms

    # --- GEMM ---
    for s in (1024, 2048, 4096, 8192):
        a = torch.randn(s, s, device=dev)
        b = torch.randn(s, s, device=dev)
        ms = timed(lambda: torch.mm(a, b))
        gflops = (2.0 * s * s * s) / (ms * 1e-3) / 1e9
        rows.append(dict(kernel="gemm_pytorch_eager", shape=f"{s}^3",
                         ours_gflops=gflops, cublas_gflops=0, pct_of_cublas=0,
                         tokens_per_s=0, max_rel_err=0))

    # --- Attention (PyTorch fused SDPA, causal) ---
    dim = 64
    for seq in (512, 1024, 2048, 4096):
        q = torch.randn(1, 1, seq, dim, device=dev)
        k = torch.randn(1, 1, seq, dim, device=dev)
        v = torch.randn(1, 1, seq, dim, device=dev)
        ms = timed(lambda: torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True))
        flops = 4.0 * seq * seq * dim
        rows.append(dict(kernel="attn_pytorch_sdpa", shape=f"seq{seq}_d{dim}",
                         ours_gflops=flops / (ms * 1e-3) / 1e9, cublas_gflops=0,
                         pct_of_cublas=0, tokens_per_s=1.0 / (ms * 1e-3),
                         max_rel_err=0))

    fields = ["kernel", "shape", "ours_gflops", "cublas_gflops",
              "pct_of_cublas", "tokens_per_s", "max_rel_err"]
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"[bench_pytorch] wrote {len(rows)} rows to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
