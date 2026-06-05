// CUDA-only helpers: error checking and lightweight event timing.
#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr)                                                      \
  do {                                                                        \
    cudaError_t _err = (expr);                                               \
    if (_err != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__,   \
                   __LINE__, cudaGetErrorString(_err));                       \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

namespace ti {

// Times a device callable (lambda taking a cudaStream_t) over `iters` runs
// after `warmup` warmups, returning the mean milliseconds per run. Uses CUDA
// events so the measurement excludes host-side launch latency amortized over
// the loop.
template <typename Fn>
float TimeKernelMs(Fn&& fn, int iters = 50, int warmup = 10,
                   cudaStream_t stream = nullptr) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int i = 0; i < warmup; ++i) fn(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < iters; ++i) fn(stream);
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / iters;
}

}  // namespace ti
