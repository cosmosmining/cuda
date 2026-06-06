#!/usr/bin/env bash
# Capture Nsight Compute profiles for each tiny-infer kernel and export both the
# raw .ncu-rep and a committed text summary into docs/profiles/.
#
#   bash scripts/profile_ncu.sh            # profile all three kernels
#   bash scripts/profile_ncu.sh gemm       # just one
#
# Requires a GPU + the CUDA toolkit (nvcc, ncu). The .ncu-rep files are
# gitignored (large/binary); the .txt summaries are committed so the profile
# story lives in the repo. See docs/profiles/README.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="docs/profiles"
mkdir -p "$OUT"

if ! command -v ncu >/dev/null 2>&1; then
  echo "error: 'ncu' (Nsight Compute) not found on PATH." >&2
  echo "Install the CUDA toolkit / Nsight Compute and re-run." >&2
  exit 1
fi

# Build the benchmarks (they double as profiling drivers).
cmake -S . -B build -DTINY_INFER_BUILD_CUDA=ON >/dev/null
cmake --build build -j >/dev/null

# Metrics that tell the perf story for each kernel:
#   - sm__throughput / dram__throughput : compute- vs memory-bound balance
#   - achieved_occupancy                : latency hiding
#   - smem bank conflicts               : shared-memory efficiency (GEMM/attn)
#   - l2 hit rate                       : reuse
SECTIONS="--section SpeedOfLight --section Occupancy \
          --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis"
# Profile only the steady-state launch (skip warmups) and one instance.
COMMON="--launch-skip 12 --launch-count 1 --target-processes all"

profile_one () {
  local name="$1" bin="$2" regex="$3"
  echo "==> Profiling $name ($regex)"
  ncu $COMMON $SECTIONS --kernel-name-base mangled --kernel-name "regex:$regex" \
      -o "$OUT/${name}" -f "./build/$bin" "data/_profile_tmp.csv"
  ncu --import "$OUT/${name}.ncu-rep" --page details > "$OUT/${name}.txt" 2>/dev/null \
    || ncu --import "$OUT/${name}.ncu-rep" > "$OUT/${name}.txt"
  echo "    wrote $OUT/${name}.ncu-rep and $OUT/${name}.txt"
}

target="${1:-all}"
[[ "$target" == "all" || "$target" == "gemm" ]] && \
  profile_one gemm_tiled       bench_gemm      "SgemmKernel"
[[ "$target" == "all" || "$target" == "attention" ]] && \
  profile_one attention_fused  bench_attention "FlashAttnKernel"
[[ "$target" == "all" || "$target" == "quant" ]] && \
  profile_one quant_int8        bench_quant     "QuantMatmulKernel"

rm -f data/_profile_tmp.csv
echo "==> Done. Commit the *.txt summaries under $OUT/."
