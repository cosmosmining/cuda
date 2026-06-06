// Fused scaled-dot-product attention (FlashAttention-1 forward).
//
//   O = softmax(Q K^T * scale + causal_mask) V,  per head, [seq, dim].
//
// The kernel never materializes the seq x seq score matrix S. Each block owns
// a BR-row query tile; one thread owns one query row. It streams BC-key tiles of
// K and V through shared memory and maintains, per query row, the online
// softmax statistics (running max m, running denominator l) and the output
// accumulator acc[HEAD_DIM] in registers, rescaling on every tile. This is the
// memory-traffic win of FlashAttention: O(seq*dim) HBM traffic instead of
// O(seq^2). The exact recurrence is unit-tested on CPU in
// tests/test_attention.cpp (AttentionOnlineCpu == AttentionCpu).
//
// HEAD_DIM is a compile-time template so q[] and acc[] live in registers with
// fully unrolled inner loops. Supported head dims: 32, 64, 128.
// Note: at HEAD_DIM=128 register pressure is high (q+acc = 256 floats/thread);
// FlashAttention-2 splits the head dim across a warp to fix this — a documented
// next step, see docs/kernels/attention.md.
#include <cfloat>

#include "cuda_utils.cuh"
#include "tiny_infer/attention.h"

namespace ti {

namespace {

constexpr int BR = 64;  // query rows per block (== threads per block)
constexpr int BC = 32;  // key/value columns streamed per tile

template <int HEAD_DIM>
__global__ __launch_bounds__(BR) void FlashAttnKernel(const float* Q,
                                                      const float* K,
                                                      const float* V, float* O,
                                                      int seq, float scale,
                                                      bool causal) {
  const int tid = threadIdx.x;             // 0..BR-1
  const int qi = blockIdx.x * BR + tid;    // global query row
  const bool active = qi < seq;

  // Shared staging for one K tile and one V tile, both [BC][HEAD_DIM].
  __shared__ float Ks[BC * HEAD_DIM];
  __shared__ float Vs[BC * HEAD_DIM];

  float q[HEAD_DIM];
  float acc[HEAD_DIM];
#pragma unroll
  for (int d = 0; d < HEAD_DIM; ++d) {
    q[d] = active ? Q[qi * HEAD_DIM + d] : 0.0f;
    acc[d] = 0.0f;
  }
  float m = -FLT_MAX;  // running row max
  float l = 0.0f;      // running row denominator

  // Per-thread inclusive key bound (causal: keys <= qi). The loop bound below is
  // block-uniform so __syncthreads() is never reached divergently.
  const int my_kmax = causal ? (active ? qi + 1 : 0) : seq;
  const int block_kmax =
      causal ? min(seq, blockIdx.x * BR + BR) : seq;  // max over the block

  for (int j0 = 0; j0 < block_kmax; j0 += BC) {
    // Cooperatively load K/V tiles (coalesced over the contiguous dim axis).
    for (int idx = tid; idx < BC * HEAD_DIM; idx += BR) {
      const int r = idx / HEAD_DIM;
      const int c = idx % HEAD_DIM;
      const int kj = j0 + r;
      const bool in = kj < seq;
      Ks[idx] = in ? K[kj * HEAD_DIM + c] : 0.0f;
      Vs[idx] = in ? V[kj * HEAD_DIM + c] : 0.0f;
    }
    __syncthreads();

    if (active) {
      const int jend = min(j0 + BC, my_kmax);
      for (int j = j0; j < jend; ++j) {
        const int r = j - j0;
        // s = scale * dot(q, K_j)
        float s = 0.0f;
#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) s += q[d] * Ks[r * HEAD_DIM + d];
        s *= scale;

        // Online softmax update.
        const float m_new = fmaxf(m, s);
        const float corr = __expf(m - m_new);  // rescale prior stats
        const float p = __expf(s - m_new);
        l = l * corr + p;
#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d)
          acc[d] = acc[d] * corr + p * Vs[r * HEAD_DIM + d];
        m = m_new;
      }
    }
    __syncthreads();  // tiles are reused next iteration
  }

  if (active) {
    const float inv = 1.0f / l;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) O[qi * HEAD_DIM + d] = acc[d] * inv;
  }
}

}  // namespace

void AttentionCudaFused(const float* dQ, const float* dK, const float* dV,
                        float* dO, int seq, int dim, float scale, bool causal,
                        void* stream) {
  dim3 block(BR);
  dim3 grid(CeilDiv(seq, BR));
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  switch (dim) {
    case 32:
      FlashAttnKernel<32><<<grid, block, 0, s>>>(dQ, dK, dV, dO, seq, scale,
                                                 causal);
      break;
    case 64:
      FlashAttnKernel<64><<<grid, block, 0, s>>>(dQ, dK, dV, dO, seq, scale,
                                                 causal);
      break;
    case 128:
      FlashAttnKernel<128><<<grid, block, 0, s>>>(dQ, dK, dV, dO, seq, scale,
                                                  causal);
      break;
    default:
      std::fprintf(stderr,
                   "AttentionCudaFused: unsupported head dim %d (use 32/64/128)\n",
                   dim);
      std::exit(EXIT_FAILURE);
  }
}

}  // namespace ti
