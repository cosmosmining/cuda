#include <cmath>
#include <vector>

#include "tiny_infer/gemm.h"
#include "tiny_infer/quant.h"
#include "test_util.h"

namespace ti {
namespace test {

namespace {
// Relative Frobenius error ||X - Y|| / ||Y||: a stable global quality metric
// for quantized matmul (per-element relative error is meaningless near zeros).
float FroRelErr(const float* x, const float* y, size_t n) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double d = double(x[i]) - double(y[i]);
    num += d * d;
    den += double(y[i]) * double(y[i]);
  }
  return float(std::sqrt(num / (den + 1e-12)));
}
}  // namespace

Stats RunQuantTests() {
  Section("INT8 quant (W8A8, per-row x per-channel)");
  Stats s;

  // --- Round-trip: dequant(quant(x)) error is bounded by half a step. ---
  {
    const int n = 1024;
    std::vector<float> x(n);
    FillRandom(x, 7, -3.0f, 3.0f);
    std::vector<int8_t> q(n);
    float scale = QuantizeAbsmax(x.data(), q.data(), n);
    float maxerr = 0.0f;
    for (int i = 0; i < n; ++i)
      maxerr = std::max(maxerr, std::fabs(x[i] - q[i] * scale));
    // Rounding error of symmetric quant is <= scale/2 (+fp slack).
    ExpectLt(s, maxerr, 0.5f * scale * 1.001f, "absmax round-trip <= step/2");
    ExpectTrue(s, scale > 0.0f, "scale positive");
  }

  // --- All-zero channel must not divide by zero. ---
  {
    std::vector<float> z(16, 0.0f);
    std::vector<int8_t> q(16);
    float scale = QuantizeAbsmax(z.data(), q.data(), 16);
    bool all_zero = true;
    for (auto v : q) all_zero &= (v == 0);
    ExpectTrue(s, scale > 0.0f && all_zero, "all-zero channel handled");
  }

  // --- End-to-end W8A8 matmul vs fp32 GEMM: error stays in the int8 regime. ---
  struct Case {
    int M, N, K;
  };
  const Case cases[] = {{128, 128, 256}, {64, 96, 512}, {33, 65, 129}};
  for (const auto& c : cases) {
    std::vector<float> A(c.M * c.K), B(c.K * c.N);
    FillRandom(A, 0xD1 + c.M, -1.0f, 1.0f);
    FillRandom(B, 0xE2 + c.N, -1.0f, 1.0f);

    std::vector<int8_t> Aq(c.M * c.K), Bq(c.K * c.N);
    std::vector<float> a_scale(c.M), b_scale(c.N);
    QuantizePerRow(A.data(), Aq.data(), a_scale.data(), c.M, c.K);
    QuantizePerChannel(B.data(), Bq.data(), b_scale.data(), c.K, c.N);

    std::vector<float> C_q(c.M * c.N), C_fp(c.M * c.N, 0.0f);
    QuantMatmulCpu(Aq.data(), Bq.data(), a_scale.data(), b_scale.data(),
                   C_q.data(), c.M, c.N, c.K);
    GemmCpu(A.data(), B.data(), C_fp.data(), c.M, c.N, c.K);

    float err = FroRelErr(C_q.data(), C_fp.data(), C_q.size());
    char name[96];
    std::snprintf(name, sizeof(name), "W8A8 vs fp32  M=%d N=%d K=%d", c.M, c.N,
                  c.K);
    // Per-channel int8 on well-conditioned data lands well under 2% Frobenius.
    ExpectLt(s, err, 2e-2, name);
  }
  return s;
}

}  // namespace test
}  // namespace ti
