// Shared benchmark utilities: device upload/download and CSV emission.
#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../src/cuda/cuda_utils.cuh"

namespace ti {
namespace bench {

template <typename T>
T* Upload(const std::vector<T>& host) {
  T* dev = nullptr;
  CUDA_CHECK(cudaMalloc(&dev, host.size() * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(dev, host.data(), host.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  return dev;
}

template <typename T>
void Download(std::vector<T>& host, const T* dev) {
  CUDA_CHECK(cudaMemcpy(host.data(), dev, host.size() * sizeof(T),
                        cudaMemcpyDeviceToHost));
}

// One row of results. `tokens_per_s` is kernel-specific (a decode-step proxy);
// leave 0 where not meaningful.
struct Row {
  std::string kernel;
  std::string shape;
  double ours_gflops = 0.0;
  double cublas_gflops = 0.0;
  double pct_of_cublas = 0.0;
  double tokens_per_s = 0.0;
  double max_rel_err = 0.0;
};

inline void AppendCsv(const std::string& path, const Row& r) {
  std::ifstream probe(path);
  const bool empty = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
  probe.close();
  std::ofstream f(path, std::ios::app);
  if (empty) {
    f << "kernel,shape,ours_gflops,cublas_gflops,pct_of_cublas,tokens_per_s,"
         "max_rel_err\n";
  }
  f << r.kernel << ',' << r.shape << ',' << r.ours_gflops << ','
    << r.cublas_gflops << ',' << r.pct_of_cublas << ',' << r.tokens_per_s << ','
    << r.max_rel_err << '\n';
}

inline void PrintRow(const Row& r) {
  std::cout << "  " << r.kernel << "  " << r.shape << "  ours="
            << r.ours_gflops << " GFLOP/s  cuBLAS=" << r.cublas_gflops
            << " GFLOP/s  (" << r.pct_of_cublas << "% of cuBLAS)  maxRelErr="
            << r.max_rel_err << "\n";
}

}  // namespace bench
}  // namespace ti
