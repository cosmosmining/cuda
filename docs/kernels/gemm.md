# Tiled SGEMM — design notes

`src/cuda/gemm_tiled.cu` · `C = alpha * A(MxK) * B(KxN) + beta * C`, row-major.

## The optimization ladder

| Step | Technique | Why it helps |
|---|---|---|
| 1 | **Block tiling** `BM×BN` per block, `BK` deep | each A/B element is read from global memory once per `BK` slab instead of once per output → ~`BK`× less global traffic |
| 2 | **Shared-memory staging** (`As`, `Bs`) | the `BM*BN` threads in a block reuse each staged element `BN`/`BM` times from fast SMEM |
| 3 | **Register tiling** `TM×TN` per thread | the inner product reads operands from registers; this is what lifts arithmetic intensity above the SMEM-bandwidth roof |
| 4 | **`float4` vectorized loads/stores** | 128-bit transactions → full DRAM/L2 transaction efficiency, fewer instructions |
| 5 | **Transposed `As` tile** (`As[k][m]`) | the inner loop reads a *column* of A as a contiguous SMEM row → conflict-free broadcast and vectorizable register fills |

## Chosen tile shape

`BM=128, BN=128, BK=8, TM=8, TN=8` → `(BM/TM)*(BN/TN) = 256` threads/block, each
computing a `TM*TN = 64`-element micro-tile.

- **Work per block:** `128×128 = 16384` outputs, `16384×K` MACs.
- **SMEM per block:** `As` `8×128` + `Bs` `8×128` = `2048` floats = **8 KB**
  (comfortably allows multiple resident blocks per SM for latency hiding).
- **Global loads per `BK` step:** `128*8 = 1024` floats per tile = exactly one
  `float4` per thread for each of A and B.
- **Arithmetic intensity** (square `N`): `2N^3 / (3N^2·4 B) = N/6` FLOP/byte →
  for `N≥1024` the kernel sits firmly in the compute-bound region of the
  [roofline](../roofline.png).

## Correctness

Cross-checked in CI against `ti::GemmCpu` (and an independent second host GEMM
with a different loop order, so a shared indexing bug can't hide) — see
`tests/test_gemm.cpp`. On the GPU, `bench_gemm` re-verifies vs the CPU reference
before timing.

## Known limitations / next steps

- **Tile alignment required:** `M%128 == 0, N%128 == 0, K%8 == 0, N%4 == 0`.
  Ragged sizes need a predicated edge kernel (standard, omitted for clarity).
- **Double buffering / `cp.async`:** overlap the next tile's global load with
  the current tile's compute (Ampere+ async copies) to hide load latency.
- **Tensor cores:** an `mma.sync` / WMMA path (bf16/tf32) is the route to
  cuBLAS-class throughput; the fp32 CUDA-core kernel here targets ~75–80% of
  cuBLAS *fp32*, not the tensor-core peak.
- **Autotuning** `BM/BN/BK/TM/TN` per architecture and shape.
