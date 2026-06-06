// CPU correctness runner. Validates the reference implementations AND the
// algorithms the GPU kernels implement (notably the online-softmax recurrence),
// so the math is verified even on a machine with no GPU. Exit code != 0 on any
// failure, so CI gates on it.
#include <cstdio>

#include "test_util.h"

namespace ti {
namespace test {
Stats RunGemmTests();
Stats RunAttentionTests();
Stats RunQuantTests();
}  // namespace test
}  // namespace ti

int main() {
  using namespace ti::test;
  std::printf("tiny-infer CPU correctness suite\n");
  Stats total;
  total.merge(RunGemmTests());
  total.merge(RunAttentionTests());
  total.merge(RunQuantTests());

  std::printf("\n----------------------------------------\n");
  std::printf("TOTAL: %d passed, %d failed\n", total.passed, total.failed);
  return total.failed == 0 ? 0 : 1;
}
