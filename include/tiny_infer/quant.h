// tiny-infer: INT8 matmul with per-channel weight scales (W8A8).
//
// Mirrors LLM weight quantization: activations A are int8 with one scale per
// row (per token), weights B are int8 with one scale per output channel
// (per column of the NxK weight, i.e. per N). The product accumulates in int32
// and dequantizes as:  C[m,n] = acc[m,n] * a_scale[m] * b_scale[n].
#pragma once

#include "tiny_infer/common.h"

namespace ti {

// Symmetric absmax quantization of one logical "channel" (a contiguous span):
// scale = max(|x|) / 127, q = round(x / scale) clamped to [-127, 127].
// Returns the scale; writes int8 codes into `out`.
float QuantizeAbsmax(const float* x, int8_t* out, int n);

// Per-row activation quant: A_fp[M,K] -> A_q[M,K] int8 + a_scale[M].
void QuantizePerRow(const float* A, int8_t* Aq, float* a_scale, int M, int K);

// Per-channel weight quant: B_fp[K,N] -> B_q[K,N] int8 + b_scale[N]
// (one scale per output column n, computed over the K elements of that column).
void QuantizePerChannel(const float* B, int8_t* Bq, float* b_scale, int K,
                        int N);

// CPU reference int8 GEMM with dequant. int32 accumulation, then per-row x
// per-channel dequantization. Ground truth for the device kernel.
void QuantMatmulCpu(const int8_t* Aq, const int8_t* Bq, const float* a_scale,
                    const float* b_scale, float* C, int M, int N, int K);

#if defined(TI_WITH_CUDA)
// Device launcher: shared-memory-blocked int8 GEMM using __dp4a (4-way int8
// dot-product into int32), then per-row x per-channel dequant on store.
// Defined in src/cuda/quant_int8_matmul.cu.
void QuantMatmulCudaInt8(const int8_t* dAq, const int8_t* dBq,
                         const float* dA_scale, const float* dB_scale,
                         float* dC, int M, int N, int K, void* stream = nullptr);
#endif

}  // namespace ti
