// tiny-infer: fused scaled-dot-product attention.
//
// O = softmax(Q K^T * scale + mask) V, per head.
// Shapes (single head, row-major): Q,K,V,O are [seq, dim]. `scale` is usually
// 1/sqrt(dim). When `causal` is true, key j > query i is masked to -inf.
#pragma once

#include "tiny_infer/common.h"

namespace ti {

// CPU reference: materializes the full S = QK^T score matrix, does a numerically
// stable row softmax, then S@V. This is the unambiguous ground truth.
void AttentionCpu(const float* Q, const float* K, const float* V, float* O,
                  int seq, int dim, float scale, bool causal);

// CPU "online-softmax" reference: computes the SAME result as AttentionCpu but
// using the streaming (running max / running sum) recurrence that the fused
// GPU kernel uses, WITHOUT ever materializing S. Testing this against
// AttentionCpu verifies the FlashAttention math on the host — no GPU required.
void AttentionOnlineCpu(const float* Q, const float* K, const float* V,
                        float* O, int seq, int dim, float scale, bool causal,
                        int block_c = 32);

#if defined(TI_WITH_CUDA)
// Device launcher: FlashAttention-style fused kernel. One query tile per block,
// streams key/value tiles through shared memory while keeping the running
// softmax statistics and output accumulator in registers; S is never written to
// global memory. Defined in src/cuda/attention_fused.cu.
void AttentionCudaFused(const float* dQ, const float* dK, const float* dV,
                        float* dO, int seq, int dim, float scale, bool causal,
                        void* stream = nullptr);
#endif

}  // namespace ti
