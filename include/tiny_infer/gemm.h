// tiny-infer: tiled SGEMM.
//
// C = alpha * A(MxK) * B(KxN) + beta * C(MxN), row-major.
#pragma once

#include "tiny_infer/common.h"

namespace ti {

// CPU reference (correctness ground truth). Always available.
// Straightforward i-k-j accumulation; not tuned for speed, tuned for being
// obviously-correct so the GPU kernel can be checked against it.
void GemmCpu(const float* A, const float* B, float* C, int M, int N, int K,
             float alpha = 1.0f, float beta = 0.0f);

#if defined(TI_WITH_CUDA)
// Device launcher: block-tiled, register-blocked SGEMM with shared-memory
// staging and float4 vectorized global loads. Defined in src/cuda/gemm_tiled.cu.
// All pointers are device pointers. Stream defaults to the null stream.
void GemmCudaTiled(const float* dA, const float* dB, float* dC, int M, int N,
                   int K, float alpha, float beta, void* stream = nullptr);
#endif

}  // namespace ti
