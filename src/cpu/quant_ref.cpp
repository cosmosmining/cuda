// CPU reference for INT8 W8A8 matmul with per-row/per-channel scales.
#include <algorithm>
#include <cmath>

#include "tiny_infer/quant.h"

namespace ti {

float QuantizeAbsmax(const float* x, int8_t* out, int n) {
  float amax = 0.0f;
  for (int i = 0; i < n; ++i) amax = std::max(amax, std::fabs(x[i]));
  // Degenerate all-zero channel: emit zeros with a unit scale (avoids div0 and
  // keeps dequant a no-op).
  const float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
  const float inv = 1.0f / scale;
  for (int i = 0; i < n; ++i) {
    int q = static_cast<int>(std::lrint(x[i] * inv));
    q = std::max(-127, std::min(127, q));  // symmetric clamp
    out[i] = static_cast<int8_t>(q);
  }
  return scale;
}

void QuantizePerRow(const float* A, int8_t* Aq, float* a_scale, int M, int K) {
  for (int m = 0; m < M; ++m) {
    a_scale[m] = QuantizeAbsmax(&A[m * K], &Aq[m * K], K);
  }
}

void QuantizePerChannel(const float* B, int8_t* Bq, float* b_scale, int K,
                        int N) {
  // Column n is strided by N in row-major [K, N] storage. Gather, quantize,
  // scatter back so the int8 weight keeps the same [K, N] layout.
  for (int n = 0; n < N; ++n) {
    float amax = 0.0f;
    for (int k = 0; k < K; ++k) amax = std::max(amax, std::fabs(B[k * N + n]));
    const float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
    const float inv = 1.0f / scale;
    for (int k = 0; k < K; ++k) {
      int q = static_cast<int>(std::lrint(B[k * N + n] * inv));
      q = std::max(-127, std::min(127, q));
      Bq[k * N + n] = static_cast<int8_t>(q);
    }
    b_scale[n] = scale;
  }
}

void QuantMatmulCpu(const int8_t* Aq, const int8_t* Bq, const float* a_scale,
                    const float* b_scale, float* C, int M, int N, int K) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      int32_t acc = 0;  // int32 accumulation, exactly as the device kernel does
      for (int k = 0; k < K; ++k) {
        acc += static_cast<int32_t>(Aq[m * K + k]) *
               static_cast<int32_t>(Bq[k * N + n]);
      }
      C[m * N + n] = static_cast<float>(acc) * a_scale[m] * b_scale[n];
    }
  }
}

}  // namespace ti
