// Benchmark: tiled SGEMM vs cuBLAS SGEMM, with a CPU-reference correctness gate.
//
//   ./bench_gemm [csv_path]
//
// Correctness is checked once at a small size against ti::GemmCpu; the sweep
// then reports sustained GFLOP/s for our kernel and cuBLAS and the achieved
// fraction of cuBLAS.
#include <cublas_v2.h>

#include <string>
#include <vector>

#include "common_bench.cuh"
#include "tiny_infer/gemm.h"

using namespace ti;
using namespace ti::bench;

#define CUBLAS_CHECK(expr)                                              \
  do {                                                                  \
    cublasStatus_t _s = (expr);                                        \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                 \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)_s,       \
                   __FILE__, __LINE__);                                 \
      std::exit(EXIT_FAILURE);                                          \
    }                                                                   \
  } while (0)

namespace {

// Row-major C[MxN] = A[MxK] * B[KxN] via column-major cuBLAS: compute C^T by
// swapping the operands (the standard row-major-on-cuBLAS idiom).
void CublasSgemmRowMajor(cublasHandle_t h, const float* dA, const float* dB,
                         float* dC, int M, int N, int K, float alpha,
                         float beta) {
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N,
                           dA, K, &beta, dC, N));
}

double GflopsPerSec(double flops, double ms) {
  return flops / (ms * 1e-3) / 1e9;
}

// Returns max relative error of our kernel vs the CPU reference at a size small
// enough to reference-check quickly.
double VerifyGemm(int M, int N, int K) {
  std::vector<float> A(M * K), B(K * N), C_ref(M * N, 0.0f), C_dev(M * N);
  FillRandom(A, 1);
  FillRandom(B, 2);
  float *dA = Upload(A), *dB = Upload(B), *dC = Upload(C_dev);
  GemmCudaTiled(dA, dB, dC, M, N, K, 1.0f, 0.0f);
  CUDA_CHECK(cudaDeviceSynchronize());
  Download(C_dev, dC);
  GemmCpu(A.data(), B.data(), C_ref.data(), M, N, K, 1.0f, 0.0f);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  return MaxRelDiff(C_dev.data(), C_ref.data(), C_ref.size(), 1e-3f);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string csv = argc > 1 ? argv[1] : "data/results.csv";
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  std::printf("== GEMM correctness (vs CPU reference) ==\n");
  double err = VerifyGemm(512, 512, 512);
  std::printf("  512^3 max rel err = %.3e  %s\n", err,
              err < 1e-2 ? "[OK]" : "[FAIL]");

  std::printf("== GEMM throughput sweep ==\n");
  const int sizes[] = {1024, 2048, 4096, 8192};
  for (int S : sizes) {
    const int M = S, N = S, K = S;
    std::vector<float> A(M * K), B(K * N), C(M * N, 0.0f);
    FillRandom(A, 1);
    FillRandom(B, 2);
    float *dA = Upload(A), *dB = Upload(B), *dC = Upload(C);

    const double flops = GemmFlops(M, N, K);
    float ours_ms = TimeKernelMs([&](cudaStream_t s) {
      GemmCudaTiled(dA, dB, dC, M, N, K, 1.0f, 0.0f, s);
    });
    float cublas_ms = TimeKernelMs([&](cudaStream_t s) {
      cublasSetStream(handle, s);
      CublasSgemmRowMajor(handle, dA, dB, dC, M, N, K, 1.0f, 0.0f);
    });

    Row r;
    r.kernel = "gemm_tiled_fp32";
    r.shape = std::to_string(S) + "^3";
    r.ours_gflops = GflopsPerSec(flops, ours_ms);
    r.cublas_gflops = GflopsPerSec(flops, cublas_ms);
    r.pct_of_cublas = 100.0 * r.ours_gflops / r.cublas_gflops;
    r.max_rel_err = (S == 1024) ? VerifyGemm(1024, 1024, 1024) : 0.0;
    PrintRow(r);
    AppendCsv(csv, r);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
  }
  cublasDestroy(handle);
  return 0;
}
