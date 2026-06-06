# Fused softmax-attention — design notes

`src/cuda/attention_fused.cu` · `O = softmax(Q K^T · scale + mask) V`, per head,
shapes `[seq, dim]`. This is a **FlashAttention-1 forward**.

## The idea: never materialize `S`

A naive implementation forms the `seq × seq` score matrix `S = QK^T`, softmaxes
its rows, then computes `S @ V`. That costs `O(seq^2)` extra HBM traffic and
memory. The fused kernel instead **streams** over key/value tiles and keeps a
running softmax, so `S` never touches global memory:

```
for each query row i (one thread):
  m = -inf; l = 0; acc[dim] = 0           # running max, denom, output
  for each key tile j0 (BC keys, staged to shared memory):
    for each key j in the tile:
      s     = scale * dot(q_i, k_j)
      m_new = max(m, s)
      corr  = exp(m - m_new)              # rescale prior partial results
      p     = exp(s - m_new)
      l     = l*corr + p
      acc   = acc*corr + p*v_j
      m     = m_new
  O_i = acc / l
```

The `exp(m - m_new)` correction is the crux: when a later tile raises the row
max, the already-accumulated `l` and `acc` are rescaled so the final result
equals a single global softmax — exactly. **This recurrence is unit-tested on
CPU**: `AttentionOnlineCpu` (the streaming form) matches `AttentionCpu` (the
textbook form) to ~`1e-7` in `tests/test_attention.cpp`, which is how the
algorithm is validated with no GPU.

## Mapping to the GPU

- **One block per query tile** (`BR=64` rows), **one thread per query row**.
- `q[HEAD_DIM]` and `acc[HEAD_DIM]` live in registers (`HEAD_DIM` is a
  compile-time template → fully unrolled inner loops). Supported dims: 32/64/128.
- Each iteration cooperatively stages a `BC=32` key tile and value tile into
  shared memory (`2 · BC · HEAD_DIM · 4 B`; 32 KB at `dim=128`), coalesced over
  the contiguous `dim` axis.
- **Causal masking** is two-tiered: a *block-uniform* upper bound on key tiles
  (so `__syncthreads()` is never reached divergently) plus a per-thread
  inclusive bound `key ≤ query`. Fully-masked tiles are skipped entirely.
- **HBM traffic** is `O(seq · dim)` (Q,K,V,O) instead of `O(seq^2)` →
  arithmetic intensity `≈ seq/4` FLOP/byte.

## Known limitations / next steps

- **Register pressure at `dim=128`:** `q+acc = 256` floats/thread pressures the
  register file and caps occupancy. **FlashAttention-2** splits the head dim
  across a warp (each lane owns part of `acc`) and swaps the loop order to fix
  this — the main throughput gap vs PyTorch's SDPA, which dispatches FA-2.
- **fp16/bf16 + tensor cores** for the `QK^T` and `PV` matmuls.
- **Multi-head / batch:** launch on `grid.y = heads*batch`; here a single head
  is exposed to match the CPU reference 1:1. Backward pass is out of scope
  (inference only).
