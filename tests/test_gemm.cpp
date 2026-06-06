#include <vector>

#include "tiny_infer/gemm.h"
#include "test_util.h"

namespace ti {
namespace test {

namespace {
// Independent naive GEMM with a different loop order than GemmCpu, so a shared
// indexing bug in the reference can't hide (the two would have to agree by luck).
void GemmNaive(const std::vector<float>& A, const std::vector<float>& B,
               std::vector<float>& C, int M, int N, int K, float alpha,
               float beta) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) C[m * N + n] *= beta;
  for (int k = 0; k < K; ++k)
    for (int m = 0; m < M; ++m) {
      const float a = alpha * A[m * K + k];
      for (int n = 0; n < N; ++n) C[m * N + n] += a * B[k * N + n];
    }
}
}  // namespace

Stats RunGemmTests() {
  Section("GEMM (CPU reference)");
  Stats s;

  struct Case {
    int M, N, K;
    float alpha, beta;
  };
  const Case cases[] = {
      {64, 48, 80, 1.0f, 0.0f},
      {128, 128, 64, 1.0f, 0.0f},
      {65, 33, 97, 0.5f, 2.0f},  // non-multiple-of-tile, with alpha/beta
  };

  for (const auto& c : cases) {
    std::vector<float> A(c.M * c.K), B(c.K * c.N), C0(c.M * c.N);
    FillRandom(A, 0x11 + c.M);
    FillRandom(B, 0x22 + c.N);
    FillRandom(C0, 0x33 + c.K);
    std::vector<float> C_ref = C0, C_naive = C0;

    GemmCpu(A.data(), B.data(), C_ref.data(), c.M, c.N, c.K, c.alpha, c.beta);
    GemmNaive(A, B, C_naive, c.M, c.N, c.K, c.alpha, c.beta);

    char name[96];
    std::snprintf(name, sizeof(name), "ref==naive  M=%d N=%d K=%d a=%.1f b=%.1f",
                  c.M, c.N, c.K, c.alpha, c.beta);
    // Two summation orders in fp32: agree to fp32 reduction noise.
    ExpectClose(s, C_ref.data(), C_naive.data(), C_ref.size(), 1e-4f, 1e-3f,
                name);
  }
  return s;
}

}  // namespace test
}  // namespace ti
