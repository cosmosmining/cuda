# tiny-infer — top-level convenience targets.
#
#   make test    build + run the CPU correctness suite (no GPU required)
#   make plots   regenerate roofline + results figures (ILLUSTRATIVE data)
#   make cuda    configure + build the CUDA kernels/benchmarks (needs nvcc)
#   make clean   remove build artifacts
#
# The CPU suite is the honesty anchor: it runs anywhere and validates the
# algorithms the GPU kernels implement. The CUDA build is delegated to CMake.

CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Iinclude -Wall -Wextra
PYTHON   ?= ./.venv/bin/python   # override with PYTHON=python3 if no venv

CPU_SRC  := src/cpu/gemm_ref.cpp src/cpu/attention_ref.cpp src/cpu/quant_ref.cpp
TEST_SRC := tests/test_gemm.cpp tests/test_attention.cpp tests/test_quant.cpp \
            tests/test_main.cpp
BIN_DIR  := tests/bin
TEST_BIN := $(BIN_DIR)/run_tests

.PHONY: test cpu plots cuda clean

test: $(TEST_BIN)
	$(TEST_BIN)

cpu: $(TEST_BIN)

$(TEST_BIN): $(CPU_SRC) $(TEST_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -Itests $(CPU_SRC) $(TEST_SRC) -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

plots:
	$(PYTHON) scripts/roofline.py
	$(PYTHON) scripts/plot_results.py

cuda:
	cmake -S . -B build -DTINY_INFER_BUILD_CUDA=ON
	cmake --build build -j

clean:
	rm -rf $(BIN_DIR) build
