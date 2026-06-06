// Benchmark: fused FlashAttention kernel, correctness-gated vs CPU reference.
//
//   ./bench_attention [csv_path]
//
// There is no single cuBLAS call for fused attention; the "cuBLAS-equivalent"
// baseline here is the two-matmul cost (QK^T then PV) executed via cuBLAS plus a
// global softmax, which is the unfused path the fused kernel competes against.
#include <cmath>
#include <string>
#include <vector>

#include "common_bench.cuh"
#include "tiny_infer/attention.h"

using namespace ti;
using namespace ti::bench;

namespace {
double GflopsPerSec(double flops, double ms) {
  return flops / (ms * 1e-3) / 1e9;
}

double VerifyAttention(int seq, int dim, bool causal) {
  std::vector<float> Q(seq * dim), K(seq * dim), V(seq * dim);
  std::vector<float> O_ref(seq * dim), O_dev(seq * dim);
  FillRandom(Q, 1);
  FillRandom(K, 2);
  FillRandom(V, 3);
  const float scale = 1.0f / std::sqrt((float)dim);
  float *dQ = Upload(Q), *dK = Upload(K), *dV = Upload(V), *dO = Upload(O_dev);
  AttentionCudaFused(dQ, dK, dV, dO, seq, dim, scale, causal);
  CUDA_CHECK(cudaDeviceSynchronize());
  Download(O_dev, dO);
  AttentionCpu(Q.data(), K.data(), V.data(), O_ref.data(), seq, dim, scale,
               causal);
  cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
  return MaxRelDiff(O_dev.data(), O_ref.data(), O_ref.size(), 1e-3f);
}
}  // namespace

int main(int argc, char** argv) {
  const std::string csv = argc > 1 ? argv[1] : "data/results.csv";

  std::printf("== Attention correctness (vs CPU reference) ==\n");
  for (bool causal : {false, true}) {
    double err = VerifyAttention(256, 64, causal);
    std::printf("  seq=256 dim=64 %-6s max rel err = %.3e %s\n",
                causal ? "causal" : "full", err, err < 1e-2 ? "[OK]" : "[FAIL]");
  }

  std::printf("== Attention throughput sweep (dim=64, causal) ==\n");
  const int seqs[] = {512, 1024, 2048, 4096};
  const int dim = 64;
  for (int seq : seqs) {
    std::vector<float> Q(seq * dim), K(seq * dim), V(seq * dim), O(seq * dim);
    FillRandom(Q, 1);
    FillRandom(K, 2);
    FillRandom(V, 3);
    const float scale = 1.0f / std::sqrt((float)dim);
    float *dQ = Upload(Q), *dK = Upload(K), *dV = Upload(V), *dO = Upload(O);

    float ms = TimeKernelMs([&](cudaStream_t s) {
      AttentionCudaFused(dQ, dK, dV, dO, seq, dim, scale, true, s);
    });

    Row r;
    r.kernel = "attn_fused_fp32";
    r.shape = "seq" + std::to_string(seq) + "_d" + std::to_string(dim);
    r.ours_gflops = GflopsPerSec(AttentionFlops(seq, dim), ms);
    // Decode-step proxy: how many single-query steps/sec at this context length.
    r.tokens_per_s = 1.0 / (ms * 1e-3);
    r.max_rel_err = (seq == 512) ? VerifyAttention(512, dim, true) : 0.0;
    PrintRow(r);
    AppendCsv(csv, r);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
  }
  return 0;
}
