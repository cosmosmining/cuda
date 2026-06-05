// Minimal dependency-free test harness (no gtest needed for the CPU suite).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>

namespace ti {
namespace test {

struct Stats {
  int passed = 0;
  int failed = 0;
  void merge(const Stats& o) {
    passed += o.passed;
    failed += o.failed;
  }
};

inline void Section(const char* name) {
  std::printf("\n=== %s ===\n", name);
}

// Assert value < bound (used for error metrics).
inline void ExpectLt(Stats& s, double value, double bound, const char* name) {
  if (value < bound) {
    s.passed++;
    std::printf("  [PASS] %-44s %.3e < %.3e\n", name, value, bound);
  } else {
    s.failed++;
    std::printf("  [FAIL] %-44s %.3e !< %.3e\n", name, value, bound);
  }
}

// NumPy-allclose-style comparison: passes iff for every element
//   |a - b| <= atol + rtol*|b|.
// The atol floor absorbs near-zero entries (where pure relative error is
// meaningless), while rtol tracks fp32 reduction noise that differs purely by
// summation order between two mathematically-equal implementations. A genuine
// algorithm bug yields a residual far above 1.0 and still fails loudly.
inline void ExpectClose(Stats& s, const float* a, const float* b, size_t n,
                        float atol, float rtol, const char* name) {
  float max_abs = 0.0f, max_res = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    const float ad = std::fabs(a[i] - b[i]);
    max_abs = std::max(max_abs, ad);
    max_res = std::max(max_res, ad / (atol + rtol * std::fabs(b[i])));
  }
  if (max_res <= 1.0f) {
    s.passed++;
    std::printf("  [PASS] %-44s maxabs=%.2e res=%.2f\n", name, max_abs, max_res);
  } else {
    s.failed++;
    std::printf("  [FAIL] %-44s maxabs=%.2e res=%.2f\n", name, max_abs, max_res);
  }
}

inline void ExpectTrue(Stats& s, bool cond, const char* name) {
  if (cond) {
    s.passed++;
    std::printf("  [PASS] %s\n", name);
  } else {
    s.failed++;
    std::printf("  [FAIL] %s\n", name);
  }
}

}  // namespace test
}  // namespace ti
