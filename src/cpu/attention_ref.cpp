// CPU reference attention.
//
// Two implementations that MUST agree bit-for-bit-ish (within fp tolerance):
//   1. AttentionCpu       - the textbook "materialize S, softmax rows, S@V".
//   2. AttentionOnlineCpu - the streaming online-softmax recurrence used by the
//                           fused FlashAttention GPU kernel.
// Their agreement is what lets us validate the kernel's algorithm on a host
// with no GPU. See tests/test_attention.cpp.
#include <limits>
#include <vector>

#include "tiny_infer/attention.h"

namespace ti {

namespace {
inline float Dot(const float* a, const float* b, int n) {
  float s = 0.0f;
  for (int i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}
}  // namespace

void AttentionCpu(const float* Q, const float* K, const float* V, float* O,
                  int seq, int dim, float scale, bool causal) {
  std::vector<float> scores(seq);
  for (int i = 0; i < seq; ++i) {
    const int jmax = causal ? (i + 1) : seq;
    // Scores + running max for numerical stability.
    float maxv = -std::numeric_limits<float>::infinity();
    for (int j = 0; j < jmax; ++j) {
      float s = scale * Dot(&Q[i * dim], &K[j * dim], dim);
      scores[j] = s;
      if (s > maxv) maxv = s;
    }
    // Exponentiate and accumulate the denominator.
    float denom = 0.0f;
    for (int j = 0; j < jmax; ++j) {
      scores[j] = std::exp(scores[j] - maxv);
      denom += scores[j];
    }
    // Weighted sum of values.
    float* o = &O[i * dim];
    for (int d = 0; d < dim; ++d) o[d] = 0.0f;
    const float inv = 1.0f / denom;
    for (int j = 0; j < jmax; ++j) {
      const float p = scores[j] * inv;
      const float* v = &V[j * dim];
      for (int d = 0; d < dim; ++d) o[d] += p * v[d];
    }
  }
}

void AttentionOnlineCpu(const float* Q, const float* K, const float* V,
                        float* O, int seq, int dim, float scale, bool causal,
                        int block_c) {
  std::vector<float> acc(dim);
  std::vector<float> s_blk(block_c);
  for (int i = 0; i < seq; ++i) {
    const int jmax = causal ? (i + 1) : seq;
    float m = -std::numeric_limits<float>::infinity();  // running max
    float l = 0.0f;                                     // running denom
    for (int d = 0; d < dim; ++d) acc[d] = 0.0f;

    for (int jb = 0; jb < jmax; jb += block_c) {
      const int jend = std::min(jb + block_c, jmax);
      // Pass 1 over the tile: scores and the tile-local max.
      float local_m = -std::numeric_limits<float>::infinity();
      for (int j = jb; j < jend; ++j) {
        float s = scale * Dot(&Q[i * dim], &K[j * dim], dim);
        s_blk[j - jb] = s;
        if (s > local_m) local_m = s;
      }
      const float m_new = std::max(m, local_m);
      // Rescale the previously accumulated statistics to the new max.
      const float corr = std::exp(m - m_new);  // exp(-inf - finite) == 0 on init
      l *= corr;
      for (int d = 0; d < dim; ++d) acc[d] *= corr;
      // Pass 2 over the tile: add this tile's contribution.
      for (int j = jb; j < jend; ++j) {
        const float p = std::exp(s_blk[j - jb] - m_new);
        l += p;
        const float* v = &V[j * dim];
        for (int d = 0; d < dim; ++d) acc[d] += p * v[d];
      }
      m = m_new;
    }

    float* o = &O[i * dim];
    const float inv = 1.0f / l;
    for (int d = 0; d < dim; ++d) o[d] = acc[d] * inv;
  }
}

}  // namespace ti
