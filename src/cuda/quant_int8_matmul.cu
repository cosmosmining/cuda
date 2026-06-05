// INT8 W8A8 matmul with per-row (activation) x per-channel (weight) scales.
//
//   C_fp[m,n] = ( sum_k A_q[m,k] * B_q[k,n] ) * a_scale[m] * b_scale[n]
//
// int8 inputs, int32 accumulation via __dp4a (one instruction does a 4-element
// int8 dot-product), fp32 dequant on store. This mirrors LLM W8A8 inference:
// activations quantized per token (row), weights per output channel (column).
//
// Both SMEM tiles are stored K-contiguous so each __dp4a reads a packed 32-bit
// word with a single aligned load:
//   As[BM][BK]  -> row m of A, contiguous over K
//   Bs[BN][BK]  -> column n of B transposed, contiguous over K
//
// Constraints (documented; harness uses aligned sizes):
//   M % BM == 0, N % BN == 0, K % BK == 0, BK % 4 == 0.
#include "cuda_utils.cuh"
#include "tiny_infer/quant.h"

namespace ti {

namespace {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;  // multiple of 4 for __dp4a
constexpr int TM = 4;   // micro-tile rows/thread
constexpr int TN = 4;   // micro-tile cols/thread
// (BM/TM) * (BN/TN) = 16 * 16 = 256 threads/block.

__global__ __launch_bounds__(256) void QuantMatmulKernel(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    const float* __restrict__ a_scale, const float* __restrict__ b_scale,
    float* __restrict__ C, int M, int N, int K) {
  const int blockRow = blockIdx.y;
  const int blockCol = blockIdx.x;
  const int threadCol = threadIdx.x % (BN / TN);  // 0..15
  const int threadRow = threadIdx.x / (BN / TN);  // 0..15

  __shared__ int8_t As[BM * BK];  // [BM][BK], K-contiguous
  __shared__ int8_t Bs[BN * BK];  // [BN][BK], K-contiguous (B transposed)

  int32_t acc[TM * TN] = {0};

  for (int k0 = 0; k0 < K; k0 += BK) {
    // Load A tile [BM][BK]: coalesced over K within each row.
    for (int idx = threadIdx.x; idx < BM * BK; idx += 256) {
      const int r = idx / BK;
      const int c = idx % BK;
      As[idx] = A[(blockRow * BM + r) * K + (k0 + c)];
    }
    // Load B tile and transpose into Bs[BN][BK]. Read B[k0+k][col] with the col
    // index contiguous across threads (coalesced GMEM); write transposed in SMEM.
    for (int idx = threadIdx.x; idx < BK * BN; idx += 256) {
      const int k = idx / BN;
      const int n = idx % BN;
      Bs[n * BK + k] = B[(k0 + k) * N + (blockCol * BN + n)];
    }
    __syncthreads();

    // Register micro-tile of int8 dot-products via __dp4a.
#pragma unroll
    for (int kk = 0; kk < BK; kk += 4) {
      int aFrag[TM];
      int bFrag[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i)
        aFrag[i] = *reinterpret_cast<const int*>(&As[(threadRow * TM + i) * BK + kk]);
#pragma unroll
      for (int j = 0; j < TN; ++j)
        bFrag[j] = *reinterpret_cast<const int*>(&Bs[(threadCol * TN + j) * BK + kk]);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i * TN + j] = __dp4a(aFrag[i], bFrag[j], acc[i * TN + j]);
    }
    __syncthreads();
  }

  // Dequantize and store: C = acc * a_scale[row] * b_scale[col].
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = blockRow * BM + threadRow * TM + i;
    const float as = a_scale[row];
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int col = blockCol * BN + threadCol * TN + j;
      C[row * N + col] = static_cast<float>(acc[i * TN + j]) * as * b_scale[col];
    }
  }
}

}  // namespace

void QuantMatmulCudaInt8(const int8_t* dAq, const int8_t* dBq,
                         const float* dA_scale, const float* dB_scale,
                         float* dC, int M, int N, int K, void* stream) {
  dim3 block(256);
  dim3 grid(CeilDiv(N, BN), CeilDiv(M, BM));
  QuantMatmulKernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      dAq, dBq, dA_scale, dB_scale, dC, M, N, K);
}

}  // namespace ti
