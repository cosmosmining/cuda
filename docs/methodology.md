# Benchmark methodology

This document describes exactly how the numbers in [`results/results.md`](results/results.md)
are produced, so they can be reproduced and trusted.

## Provenance of the numbers in this repo

> **The committed `data/results.csv` and every figure are ILLUSTRATIVE
> placeholders**, not measurements. They exist to exercise the plotting and
> docs pipeline and to show the intended layout. Each figure is watermarked
> `ILLUSTRATIVE`, and `data/results.csv` carries a `SYNTHETIC PLACEHOLDER`
> banner. Running `bash bench/run_all.sh` on a real GPU **deletes and rewrites**
> `data/results.csv` from the correctness-gated benchmarks (without the banner),
> after which the figures regenerate as real measurements.
>
> What *is* real today: the kernels themselves, the CPU reference
> implementations, and the [CPU correctness suite](../tests) — which validates
> the kernel algorithms (including the FlashAttention online-softmax recurrence)
> and runs in CI on every commit.

## What each kernel is compared against

| Kernel | Our implementation | Baseline | Correctness reference |
|---|---|---|---|
| `gemm_tiled_fp32` | block + register tiled SGEMM | cuBLAS `cublasSgemm` | `ti::GemmCpu` |
| `attn_fused_fp32` | FlashAttention-1 forward | PyTorch SDPA (FA-2) | `ti::AttentionCpu` |
| `matmul_int8` | `__dp4a` tiled int8 GEMM | cuBLAS `cublasGemmEx` (int8) | `ti::QuantMatmulCpu` / fp32 GEMM |

## Timing

- CUDA events bracket a loop of `iters` launches after `warmup` launches; the
  reported time is the mean per-launch millisecond figure
  (`ti::TimeKernelMs`, `src/cuda/cuda_utils.cuh`). Warmups absorb one-time JIT,
  clock ramp, and allocator effects.
- PyTorch baselines use `torch.cuda.synchronize()` around an equivalent loop
  (`bench/bench_pytorch.py`) with TF32 **disabled** so the comparison is true
  fp32 against true fp32.
- Clocks: for stable numbers, lock clocks before profiling, e.g.
  `sudo nvidia-smi -lgc <clock>` and record `nvidia-smi -q` in your run log.

## Throughput formulas

Defined once in `include/tiny_infer/common.h` so host and device agree:

- **GEMM**: `2 * M * N * K` FLOPs.
- **Attention** (per head): `4 * S^2 * d` FLOPs — the two matmuls `QK^T` and
  `PV`; the softmax is excluded from the headline FLOP count (it is
  bandwidth/latency bound, not flop bound).
- **INT8 matmul**: `2 * M * N * K` int8 MACs, reported as OP/s.
- `GFLOP/s = FLOPs / (ms * 1e-3) / 1e9`. "% of cuBLAS" = `ours / cublas * 100`.

## Correctness gate

Every benchmark **verifies before it times**. The device result is compared to
the CPU reference; GEMM/attention use a NumPy-`allclose`-style
`|a-b| <= atol + rtol*|b|` test at fp32 tolerances, and the int8 path uses a
Frobenius relative error bounded by the int8 quantization regime (~5e-3 on
well-conditioned data). A failing gate aborts the run — we never report a
throughput number for a kernel that did not match the reference.

## Hardware to record per run

Fill these in alongside any results you commit (see the table header in
`results/results.md`):

- GPU model + SKU (e.g. *H100 SXM5 80GB*), driver, CUDA toolkit version.
- Locked SM/memory clocks (or "default/auto").
- Host: CPU, the cuBLAS and PyTorch versions used as baselines.
- For the PSC **Bridges-2** runs: partition/node type and the module stack.
