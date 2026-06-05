#!/usr/bin/env bash
# Build the CUDA benchmarks and run the full suite, writing a fresh results CSV
# and regenerating the figures. Run from the repo root on a GPU machine.
#
#   bash bench/run_all.sh
#
# Output: data/results.csv  +  docs/results/*.png  +  docs/roofline.png
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CSV="data/results.csv"
PYTHON="${PYTHON:-python3}"

echo "==> Building CUDA kernels + benchmarks"
cmake -S . -B build -DTINY_INFER_BUILD_CUDA=ON
cmake --build build -j

echo "==> Starting fresh results CSV"
mkdir -p data
rm -f "$CSV"

echo "==> Running CUDA benchmarks (correctness-gated)"
./build/bench_gemm        "$CSV"
./build/bench_attention   "$CSV"
./build/bench_quant       "$CSV"

echo "==> Running PyTorch eager baselines (skips if no CUDA torch)"
"$PYTHON" bench/bench_pytorch.py "$CSV" || true

echo "==> Regenerating figures"
"$PYTHON" scripts/plot_results.py
"$PYTHON" scripts/roofline.py

echo "==> Done. See $CSV and docs/."
