# Nsight Compute profiles

This directory holds the committed **text summaries** of Nsight Compute (`ncu`)
profiles for each kernel. The raw `*.ncu-rep` files are gitignored (large,
binary); regenerate them with:

```bash
bash scripts/profile_ncu.sh            # all three kernels
bash scripts/profile_ncu.sh gemm       # one kernel
```

That script builds the benchmarks, runs `ncu` on the steady-state launch of each
kernel, and writes `<kernel>.ncu-rep` (raw) + `<kernel>.txt` (committed summary)
here.

> **Status:** no profiles are committed yet — they require a GPU + the CUDA
> toolkit, which the authoring environment did not have. The capture pipeline
> and the metric selection below are ready; running `scripts/profile_ncu.sh` on
> an H100 populates this directory. Nothing here is fabricated.

## Metrics captured (and why)

The script requests these `ncu` sections — the ones that tell each kernel's
perf story:

| Section | Reads on | What it answers |
|---|---|---|
| **SpeedOfLight** | SM% vs DRAM% | compute-bound or memory-bound? where on the roofline? |
| **Occupancy** | achieved vs theoretical | is latency being hidden; is occupancy register/SMEM-limited? |
| **MemoryWorkloadAnalysis** | L1/L2 hit rates, SMEM bank conflicts | is shared-memory staging conflict-free; is reuse landing in L2? |
| **ComputeWorkloadAnalysis** | pipe utilization (FMA / `dp4a` / tensor) | which math pipe is the bottleneck |

## What to look for per kernel

- **`gemm_tiled` (`SgemmKernel`):** high SM%, near-zero SMEM bank conflicts (the
  transposed `As` tile), FMA pipe as the limiter. Gap to cuBLAS shows up as
  occupancy or memory-latency stalls → motivates `cp.async` double buffering.
- **`attention_fused` (`FlashAttnKernel`):** watch achieved occupancy at
  `dim=128` (register-pressure limited — the FA-2 motivation) and confirm DRAM%
  is low (the fused `O(seq·dim)` traffic win).
- **`quant_int8` (`QuantMatmulKernel`):** expect the **`dp4a`/INT pipe**
  saturated well below the int8 tensor-core ceiling — the quantitative case for
  an `mma.sync` tensor-core path.
