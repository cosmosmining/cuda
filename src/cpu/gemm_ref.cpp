// CPU reference SGEMM. Correctness ground truth for the tiled CUDA kernel.
#include "tiny_infer/gemm.h"

namespace ti {

void GemmCpu(const float* A, const float* B, float* C, int M, int N, int K,
             float alpha, float beta) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += A[m * K + k] * B[k * N + n];
      }
      const int idx = m * N + n;
      C[idx] = alpha * acc + beta * C[idx];
    }
  }
}

}  // namespace ti
