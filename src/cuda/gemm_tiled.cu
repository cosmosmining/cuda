// Tiled SGEMM: C = alpha * A(MxK) * B(KxN) + beta * C, row-major.
//
// Optimization ladder implemented here (each step raises arithmetic intensity
// or memory efficiency; see docs/kernels/gemm.md for the full write-up):
//   1. Block tiling      - each block computes a BM x BN output tile, reading
//                          A/B tiles from global memory only once per BK slab,
//                          cutting global traffic by ~BK vs the naive kernel.
//   2. Shared-memory     - A/B tiles are staged in SMEM so the BM*BN threads
//                          reuse each loaded element BN / BM times respectively.
//   3. Register tiling   - each thread computes a TM x TN micro-tile held in
//                          registers, so the inner product reuses operands from
//                          registers (the real throughput lever).
//   4. float4 loads      - global loads/stores are 128-bit vectorized for full
//                          memory-transaction efficiency.
//   5. Transposed A tile - As is stored [BK][BM] so the inner loop reads a
//                          column of A as a contiguous SMEM row -> no bank
//                          conflicts and enables float4 register fills.
//
// Constraints (documented; the benchmark harness uses tile-aligned sizes):
//   M % BM == 0, N % BN == 0, K % BK == 0, and N % 4 == 0 for vectorization.
// Production code adds a predicated edge kernel for ragged sizes.
#include "cuda_utils.cuh"
#include "tiny_infer/gemm.h"

namespace ti {

namespace {

constexpr int BM = 128;  // output tile rows per block
constexpr int BN = 128;  // output tile cols per block
constexpr int BK = 8;    // contraction-dim slab
constexpr int TM = 8;    // micro-tile rows per thread
constexpr int TN = 8;    // micro-tile cols per thread
// (BM/TM) * (BN/TN) = 16 * 16 = 256 threads/block, each owning TM*TN = 64 outs.

__global__ __launch_bounds__(256) void SgemmKernel(int M, int N, int K,
                                                   float alpha, const float* A,
                                                   const float* B, float beta,
                                                   float* C) {
  const int blockRow = blockIdx.y;
  const int blockCol = blockIdx.x;

  // This thread's micro-tile origin within the block's BM x BN output tile.
  const int threadCol = threadIdx.x % (BN / TN);  // 0..15
  const int threadRow = threadIdx.x / (BN / TN);  // 0..15

  __shared__ float As[BK * BM];  // transposed: As[k*BM + m]
  __shared__ float Bs[BK * BN];  // Bs[k*BN + n]

  // Advance global pointers to this block's tiles.
  A += blockRow * BM * K;
  B += blockCol * BN;
  C += blockRow * BM * N + blockCol * BN;

  // Vectorized (float4) load mappings. 256 threads move 128*8 = 1024 floats per
  // tile = 256 float4s, i.e. exactly one float4 per thread.
  const int innerRowA = threadIdx.x / (BK / 4);  // 0..127  (BK/4 = 2)
  const int innerColA = threadIdx.x % (BK / 4);  // 0..1
  const int innerRowB = threadIdx.x / (BN / 4);  // 0..7    (BN/4 = 32)
  const int innerColB = threadIdx.x % (BN / 4);  // 0..31

  float acc[TM * TN] = {0.0f};
  float regM[TM];
  float regN[TN];

  for (int k0 = 0; k0 < K; k0 += BK) {
    // Load A tile and transpose into SMEM (column of A -> row of As).
    float4 a = reinterpret_cast<const float4*>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = a.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = a.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = a.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = a.w;

    // Load B tile directly (already row-major friendly).
    reinterpret_cast<float4*>(&Bs[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4*>(&B[innerRowB * N + innerColB * 4])[0];

    __syncthreads();
    A += BK;
    B += BK * N;

    // Compute the BK-deep outer products into the register micro-tile.
#pragma unroll
    for (int dot = 0; dot < BK; ++dot) {
#pragma unroll
      for (int i = 0; i < TM; ++i) regM[i] = As[dot * BM + threadRow * TM + i];
#pragma unroll
      for (int j = 0; j < TN; ++j) regN[j] = Bs[dot * BN + threadCol * TN + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i * TN + j] += regM[i] * regN[j];
    }
    __syncthreads();
  }

  // Epilogue: C = alpha*acc + beta*C, written as float4.
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = threadRow * TM + i;
#pragma unroll
    for (int j = 0; j < TN; j += 4) {
      const int col = threadCol * TN + j;
      float4 c = reinterpret_cast<float4*>(&C[row * N + col])[0];
      c.x = alpha * acc[i * TN + j + 0] + beta * c.x;
      c.y = alpha * acc[i * TN + j + 1] + beta * c.y;
      c.z = alpha * acc[i * TN + j + 2] + beta * c.z;
      c.w = alpha * acc[i * TN + j + 3] + beta * c.w;
      reinterpret_cast<float4*>(&C[row * N + col])[0] = c;
    }
  }
}

}  // namespace

void GemmCudaTiled(const float* dA, const float* dB, float* dC, int M, int N,
                   int K, float alpha, float beta, void* stream) {
  dim3 block(256);
  dim3 grid(CeilDiv(N, BN), CeilDiv(M, BM));
  SgemmKernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      M, N, K, alpha, dA, dB, beta, dC);
}

}  // namespace ti
