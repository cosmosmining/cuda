// Benchmark: INT8 W8A8 matmul vs cuBLAS int8 GEMM (cublasGemmEx), correctness
// gated vs the CPU reference.
//
//   ./bench_quant [csv_path]
//
// cuBLAS baseline uses CUDA_R_8I inputs, CUDA_R_32I output, COMPUTE_32I (raw
// int8 GEMM, no dequant). Our kernel additionally fuses per-row x per-channel
// dequant, so this is a slightly conservative comparison for us.
#include <cublas_v2.h>

#include <string>
#include <vector>

#include "common_bench.cuh"
#include "tiny_infer/gemm.h"
#include "tiny_infer/quant.h"

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
double GopsPerSec(double ops, double ms) { return ops / (ms * 1e-3) / 1e9; }

// Verify our int8 kernel (with dequant) against the fp32 GEMM of the original
// values: error must stay within the int8 quantization regime.
double VerifyQuant(int M, int N, int K) {
  std::vector<float> A(M * K), B(K * N), C_fp(M * N, 0.0f), C_dev(M * N);
  FillRandom(A, 1, -1.0f, 1.0f);
  FillRandom(B, 2, -1.0f, 1.0f);
  std::vector<int8_t> Aq(M * K), Bq(K * N);
  std::vector<float> as(M), bs(N);
  QuantizePerRow(A.data(), Aq.data(), as.data(), M, K);
  QuantizePerChannel(B.data(), Bq.data(), bs.data(), K, N);

  int8_t *dAq = Upload(Aq), *dBq = Upload(Bq);
  float *das = Upload(as), *dbs = Upload(bs), *dC = Upload(C_dev);
  QuantMatmulCudaInt8(dAq, dBq, das, dbs, dC, M, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());
  Download(C_dev, dC);
  GemmCpu(A.data(), B.data(), C_fp.data(), M, N, K);
  cudaFree(dAq); cudaFree(dBq); cudaFree(das); cudaFree(dbs); cudaFree(dC);

  // Frobenius relative error is the right global metric for quantized output.
  double num = 0, den = 0;
  for (size_t i = 0; i < C_fp.size(); ++i) {
    double d = C_dev[i] - C_fp[i];
    num += d * d;
    den += (double)C_fp[i] * C_fp[i];
  }
  return std::sqrt(num / (den + 1e-12));
}
}  // namespace

int main(int argc, char** argv) {
  const std::string csv = argc > 1 ? argv[1] : "data/results.csv";
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  std::printf("== INT8 matmul correctness (W8A8 vs fp32 GEMM) ==\n");
  double e = VerifyQuant(512, 512, 512);
  std::printf("  512^3 Frobenius rel err = %.3e %s\n", e,
              e < 2e-2 ? "[OK]" : "[FAIL]");

  std::printf("== INT8 matmul throughput sweep ==\n");
  const int sizes[] = {1024, 2048, 4096, 8192};
  for (int S : sizes) {
    const int M = S, N = S, K = S;
    std::vector<int8_t> Aq(M * K, 1), Bq(K * N, 1);
    std::vector<float> as(M, 1.0f), bs(N, 1.0f), C(M * N);
    int8_t *dAq = Upload(Aq), *dBq = Upload(Bq);
    float *das = Upload(as), *dbs = Upload(bs), *dC = Upload(C);
    int32_t* dCi = nullptr;
    CUDA_CHECK(cudaMalloc(&dCi, (size_t)M * N * sizeof(int32_t)));

    const double ops = GemmFlops(M, N, K);  // 2*M*N*K int8 MACs
    float ours_ms = TimeKernelMs([&](cudaStream_t s) {
      QuantMatmulCudaInt8(dAq, dBq, das, dbs, dC, M, N, K, s);
    });
    const int32_t ialpha = 1, ibeta = 0;
    float cublas_ms = TimeKernelMs([&](cudaStream_t s) {
      cublasSetStream(handle, s);
      cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &ialpha, dBq,
                   CUDA_R_8I, N, dAq, CUDA_R_8I, K, &ibeta, dCi, CUDA_R_32I, N,
                   CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    });

    Row r;
    r.kernel = "matmul_int8";
    r.shape = std::to_string(S) + "^3";
    r.ours_gflops = GopsPerSec(ops, ours_ms);      // GOP/s (int8)
    r.cublas_gflops = GopsPerSec(ops, cublas_ms);
    r.pct_of_cublas = 100.0 * r.ours_gflops / r.cublas_gflops;
    r.max_rel_err = (S == 1024) ? VerifyQuant(1024, 1024, 1024) : 0.0;
    PrintRow(r);
    AppendCsv(csv, r);
    cudaFree(dAq); cudaFree(dBq); cudaFree(das); cudaFree(dbs);
    cudaFree(dC); cudaFree(dCi);
  }
  cublasDestroy(handle);
  return 0;
}
