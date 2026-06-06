// tiny-infer: shared host-side utilities.
//
// This header is intentionally free of any CUDA dependency so it compiles
// cleanly with a plain host compiler (g++/clang++) for the CPU reference and
// correctness tests, and is equally usable from .cu translation units.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace ti {

// Round up `a` to the next multiple of `b` (b > 0). Ubiquitous for grid sizing.
inline int CeilDiv(int a, int b) { return (a + b - 1) / b; }

// Deterministic fill in [lo, hi). A fixed seed keeps tests reproducible and
// makes ref-vs-kernel comparisons bit-stable across runs.
inline void FillRandom(std::vector<float>& v, uint64_t seed, float lo = -1.0f,
                       float hi = 1.0f) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (auto& x : v) x = dist(rng);
}

// Max absolute difference between two equally sized buffers.
inline float MaxAbsDiff(const float* a, const float* b, size_t n) {
  float m = 0.0f;
  for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

// Max relative difference with an absolute floor, robust to values near zero.
// rel = |a-b| / max(|a|, |b|, eps). This is the metric the tests assert on.
inline float MaxRelDiff(const float* a, const float* b, size_t n,
                        float eps = 1e-4f) {
  float m = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float denom = std::max({std::fabs(a[i]), std::fabs(b[i]), eps});
    m = std::max(m, std::fabs(a[i] - b[i]) / denom);
  }
  return m;
}

// Theoretical FLOP counts used by both the benchmark harness and the roofline
// script, kept here so host and device code agree on the arithmetic.
//   GEMM: 2*M*N*K (one multiply + one add per inner-product term).
inline double GemmFlops(int M, int N, int K) {
  return 2.0 * M * N * K;
}

// Attention O = softmax(QK^T/sqrt(d)) V for one head:
//   QK^T : 2*S*S*d, softmax ~ a few S*S ops, (.)V : 2*S*S*d.
// We report the two matmuls (the dominant term); the softmax is bandwidth-,
// not flop-, bound and is excluded from the headline FLOP count.
inline double AttentionFlops(int seq, int dim) {
  return 4.0 * static_cast<double>(seq) * seq * dim;
}

}  // namespace ti
