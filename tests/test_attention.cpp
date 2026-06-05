#include <cmath>
#include <vector>

#include "tiny_infer/attention.h"
#include "test_util.h"

namespace ti {
namespace test {

Stats RunAttentionTests() {
  Section("Attention (full vs online-softmax)");
  Stats s;

  // --- Anchored case: hand-computed expected output pins absolute correctness,
  // not just self-consistency between the two CPU paths. ---
  {
    const int seq = 2, dim = 2;
    const float Q[] = {1, 0, 0, 1};
    const float K[] = {1, 0, 0, 1};
    const float V[] = {1, 2, 3, 4};
    // softmax([1,0]) = [0.73105858, 0.26894142]; row0 = p0*V0 + p1*V1, etc.
    const float expected[] = {1.53788284f, 2.53788284f, 2.46211716f,
                              3.46211716f};
    std::vector<float> O(seq * dim);
    AttentionCpu(Q, K, V, O.data(), seq, dim, 1.0f, /*causal=*/false);
    ExpectLt(s, MaxAbsDiff(O.data(), expected, O.size()), 1e-4,
             "anchored 2x2 == hand-computed");
  }

  // --- Streaming math == textbook math, across shapes / masks / tile sizes. ---
  struct Case {
    int seq, dim, block_c;
    bool causal;
  };
  const Case cases[] = {
      {128, 64, 32, false}, {128, 64, 32, true},  {200, 48, 16, false},
      {200, 48, 64, true},  {37, 40, 8, true},    {256, 128, 128, false},
  };
  for (const auto& c : cases) {
    std::vector<float> Q(c.seq * c.dim), K(c.seq * c.dim), V(c.seq * c.dim);
    FillRandom(Q, 0xA1 + c.seq);
    FillRandom(K, 0xB2 + c.dim);
    FillRandom(V, 0xC3 + c.block_c);
    const float scale = 1.0f / std::sqrt(static_cast<float>(c.dim));

    std::vector<float> O_full(c.seq * c.dim), O_online(c.seq * c.dim);
    AttentionCpu(Q.data(), K.data(), V.data(), O_full.data(), c.seq, c.dim,
                 scale, c.causal);
    AttentionOnlineCpu(Q.data(), K.data(), V.data(), O_online.data(), c.seq,
                       c.dim, scale, c.causal, c.block_c);

    char name[96];
    std::snprintf(name, sizeof(name),
                  "online==full seq=%d dim=%d Bc=%d %s", c.seq, c.dim,
                  c.block_c, c.causal ? "causal" : "full");
    // Streaming softmax rescaling vs one-shot: fp32-equal up to reduction noise.
    ExpectClose(s, O_full.data(), O_online.data(), O_full.size(), 1e-4f, 1e-3f,
                name);
  }

  // --- Property: causal masking must actually change the result. ---
  {
    const int seq = 64, dim = 32;
    std::vector<float> Q(seq * dim), K(seq * dim), V(seq * dim);
    FillRandom(Q, 1);
    FillRandom(K, 2);
    FillRandom(V, 3);
    const float scale = 1.0f / std::sqrt((float)dim);
    std::vector<float> O_full(seq * dim), O_causal(seq * dim);
    AttentionCpu(Q.data(), K.data(), V.data(), O_full.data(), seq, dim, scale,
                 false);
    AttentionCpu(Q.data(), K.data(), V.data(), O_causal.data(), seq, dim, scale,
                 true);
    ExpectTrue(s, MaxAbsDiff(O_full.data(), O_causal.data(), O_full.size()) >
                      1e-3,
               "causal mask changes output");
  }
  return s;
}

}  // namespace test
}  // namespace ti
