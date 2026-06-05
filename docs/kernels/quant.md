# INT8 quantized matmul — design notes

`src/cuda/quant_int8_matmul.cu` · W8A8 inference matmul with per-row activation
scales and per-channel weight scales.

## Quantization scheme

Symmetric absmax int8, matching common LLM W8A8 inference:

```
scale   = max(|x|) / 127
q       = clamp(round(x / scale), -127, 127)          # int8 code
dequant = q * scale
```

- **Activations `A[M,K]`**: one scale **per row** (per token) — `a_scale[M]`.
- **Weights `B[K,N]`**: one scale **per output channel** (per column `n`,
  computed over that column's `K` entries) — `b_scale[N]`. Per-channel is the
  standard accuracy lever for weight quantization.
- **Dequant after the int32 accumulation:**
  `C[m,n] = acc[m,n] · a_scale[m] · b_scale[n]`.

Quant/dequant helpers and the CPU reference matmul are in `src/cpu/quant_ref.cpp`
and exercised by `tests/test_quant.cpp` (round-trip error ≤ ½ step; end-to-end
W8A8 vs fp32 GEMM Frobenius error ~`5.5e-3` on well-conditioned data).

## The device kernel

- `BM=64, BN=64, BK=16`, `TM=TN=4` → 256 threads/block, int32 register
  accumulators.
- **`__dp4a`** does a 4-way int8 dot-product into int32 in one instruction. For
  it to issue a single aligned 32-bit load, **both** SMEM tiles are stored
  **K-contiguous**:
  - `As[BM][BK]` — row `m` of A, contiguous over K (natural layout).
  - `Bs[BN][BK]` — column `n` of B **transposed** into a contiguous K run. The
    global read of `B[k][n]` is kept coalesced (the `n` index is contiguous
    across threads); the transpose happens on the SMEM write.
- Per-row × per-channel **dequant is fused into the store** — no separate pass.

## Honest performance expectation

`__dp4a` runs on the **INT32 ALU pipes**, *not* the INT8 **tensor cores**. cuBLAS
`cublasGemmEx` int8 uses tensor cores and is several× faster. So this kernel is
expected to reach only a modest fraction of cuBLAS int8 (the illustrative
placeholder shows ~12%). That gap is the point of the next step:

## Known limitations / next steps

- **`mma.sync` int8 tensor-core path** (`m16n8k32`) is the route to cuBLAS-class
  int8 throughput; the `__dp4a` kernel here is the portable, readable baseline.
- **Alignment required:** `M%64, N%64, K%16 == 0`, `BK%4 == 0`.
- **Vectorized SMEM loads** (`int4` / 16-byte) and double buffering.
- **INT8 with zero-points** (asymmetric) and group-wise weight scales (e.g.
  AWQ/GPTQ group size 128) for tighter accuracy.
