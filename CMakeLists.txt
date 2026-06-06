cmake_minimum_required(VERSION 3.18)
project(tiny_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TINY_INFER_BUILD_CUDA "Build the CUDA kernels and benchmarks" OFF)

include_directories(${CMAKE_SOURCE_DIR}/include)

# ---------------------------------------------------------------------------
# CPU correctness suite (always available; no GPU required).
# ---------------------------------------------------------------------------
add_executable(run_tests
  src/cpu/gemm_ref.cpp
  src/cpu/attention_ref.cpp
  src/cpu/quant_ref.cpp
  tests/test_gemm.cpp
  tests/test_attention.cpp
  tests/test_quant.cpp
  tests/test_main.cpp)
target_include_directories(run_tests PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_compile_options(run_tests PRIVATE -O2 -Wall -Wextra)

enable_testing()
add_test(NAME cpu_correctness COMMAND run_tests)

# ---------------------------------------------------------------------------
# CUDA kernels + cuBLAS benchmarks (opt-in: -DTINY_INFER_BUILD_CUDA=ON).
# ---------------------------------------------------------------------------
if(TINY_INFER_BUILD_CUDA)
  enable_language(CUDA)
  # Default to common datacenter/workstation arches incl. Hopper (sm_90).
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
  endif()
  find_package(CUDAToolkit REQUIRED)

  # Kernels + CPU references compiled into one static lib the benches link.
  add_library(tiny_infer_kernels STATIC
    src/cuda/gemm_tiled.cu
    src/cuda/attention_fused.cu
    src/cuda/quant_int8_matmul.cu
    src/cpu/gemm_ref.cpp
    src/cpu/attention_ref.cpp
    src/cpu/quant_ref.cpp)
  target_compile_definitions(tiny_infer_kernels PUBLIC TI_WITH_CUDA)
  set_target_properties(tiny_infer_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

  foreach(bench bench_gemm bench_attention bench_quant)
    add_executable(${bench} bench/${bench}.cu)
    target_compile_definitions(${bench} PRIVATE TI_WITH_CUDA)
    target_link_libraries(${bench} PRIVATE tiny_infer_kernels CUDA::cublas)
  endforeach()
endif()
