# Summary of the AI session (what was done, in order)

1. **Inspect the environment.** 4-core Intel Xeon (Sapphire Rapids) VM,
   15 GiB RAM, gcc 13 with OpenMP. Chose C11 + OpenMP.
2. **Fix the definition.** out[C(i)] = in[F(i)], i.e. Fortran -> C;
   C -> Fortran is the same operation on the reversed shape. Checked it
   against the handout's A[K][M][N] -> C[N][M][K] example.
3. **Implement `reorder.c`:**
   * index recalculation (div/mod; shift/mask for power-of-two extents);
   * iterative odometer;
   * recursion on a flat buffer, R = (I ⊗ R')·L with L folded into the
     strides;
   * the same recursion on an n-d strided view type;
   * FFT-style n-1 stride-permutation passes;
   * cache-blocked tiles.
4. **Write the test suite (`main.c test`).**
   * Independent reference written from the definition.
   * Native C 2-d/3-d/4-d array checks, including the handout example.
   * Bit reversal for ranks 1-20.
   * 400 random shapes (extents including 1).
   * Every square shape of 2^24.
   * Non-power-of-two shapes.
   * Thread counts 1, 2, 3, 4 and 7.
   * Byte-plane labels, so that equal byte values cannot hide a wrong
     permutation.
   * Round trip C -> F.

   First version was too slow (it recomputed the reference per algorithm);
   restructured. Result: 2631 checks pass for all six algorithms.
5. **Mutation-test the suite.** Four bugs were injected (odometer carry,
   wrong stride table in blocked, range rounding, ping-pong parity); every
   one was caught, the first by a segfault.
6. **First 2^32 measurements:**
   * unblocked reorders: ~0.4 GB/s at 4 cores;
   * plain copy (rank 1): ~65 GB/s;
   * first blocked version: ~1.7 GB/s.
7. **Diagnose blocked.** Transparent huge pages were confirmed active, so
   the TLB was ruled out. The cause was set conflicts: tile rows are 2^m
   bytes apart and map to one cache set. Buffering the tile in L1 gave ~4x.
   SSE2 16x16 in-register transposes and streaming stores were then added
   as kernel levels 2-3 (`-DBLOCK_KERNEL`). All four builds pass the test
   suite.
8. **Write scripts.** `run_all.sh` does build -> test -> bench (1..p cores)
   -> size sweep -> Part 3 ladder -> plots. `plots.py` writes Excel charts
   with openpyxl and a PDF with matplotlib. `package.sh` builds the zip.
9. **Run everything on the development VM**; wrote the README and the text
   answers.
