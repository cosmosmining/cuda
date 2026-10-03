# 18-647 HW2: rank-N tensor conversion, Fortran ↔ C ordering

Language: **C11 + OpenMP** (shared-memory parallelism). The only dependency is a C compiler with OpenMP, for example `gcc` ≥ 7, which ships `libgomp`. The plots need Python 3 with `openpyxl` and `matplotlib` (see `requirements.txt`).

## How to run

```bash
cd src
make                 # builds ./reorder  (gcc -O3 -march=native -fopenmp)
./reorder test       # correctness suite: all 6 implementations, 1..7 threads (~1 min)
./run_all.sh         # full measurement -> ../runs/*.txt, ../plots.xlsx, ../plots.pdf
```

`run_all.sh` needs about 9 GiB of RAM for the 2³²-byte cube: 4 GiB of input plus 4 GiB of output. The optional `stages` variant needs a third 4 GiB buffer and is skipped automatically if memory is short. On 16 cores the full run takes about 1–2 hours, so start it inside `tmux`. You can tune the run with these environment variables:

| variable | default | meaning |
|---|---|---|
| `CORES` | all physical cores | scale up to this many cores |
| `THREADS` | `1-$CORES` | explicit list, for example `1,2,4,8,16` |
| `REPS` | `3` | repetitions per configuration; the best one is reported |
| `LOG2N` | `32` | cube size 2^LOG2N bytes; use 30 on a small laptop |

To run single benchmarks by hand:

```bash
./reorder bench -a iterative -t 1-16 -r 3            # main measurement
./reorder bench -a all -t 16 -r 1 -k 2,32            # every algorithm, ranks 2 and 32
./reorder bench -a blocked -l 24,28,32 -t 16         # several problem sizes
```

Each bench line reports the min, median and max time and the bandwidth 2N/t_min. A sampled check of 2²⁰ random positions against the definition runs after every configuration, and every line is followed by `CSV,…` records that `plots.py` reads.

## Files

| file | content |
|---|---|
| `reorder.h` | API: `reorder_f2c(alg, in, out, rank, dims, tmp)` and `reorder_c2f(...)` |
| `reorder.c` | the six implementations (below) |
| `main.c` | test suite and benchmark driver (timing with `gettimeofday()`) |
| `run_all.sh` | build, test, measure and plot in one script |
| `plots.py` | builds `plots.xlsx` (native Excel charts) and `plots.pdf` from `../runs` |

## Definition

The tensor has shape (d₀,…,d_{n−1}) and N = ∏dₖ one-byte elements.

* The Fortran offset is F(i) = i₀ + d₀(i₁ + d₁(i₂ + …)), so i₀ is the fastest index.
* The C offset is C(i) = i_{n−1} + d_{n−1}(i_{n−2} + …), so i_{n−1} is the fastest index.

`reorder_f2c` computes out[C(i)] = in[F(i)] for every multi-index i. `reorder_c2f` is the inverse; it equals `f2c` applied to the reversed shape.

This matches the handout. A buffer B holding the C array `A[K][M][N]` is the Fortran layout of shape (N,M,K). `f2c` turns it into the C array `C[N][M][K]` with `C[k][j][i] == A[i][j][k]`, and the test suite checks exactly this with native C arrays. With all dₖ = 2 the operation is FFT bit reversal, and the tests check that too.

## The three approaches (plus two extras)

| `-a` name | approach | idea | passes |
|---|---|---|---|
| `index` | **index recalculation** | for every output position j, decode its digits (C order) and re-encode them with the Fortran strides. With power-of-two extents the digits are bit fields, so the decode uses shift and mask; with all extents 2 this is FFT bit reversal | 1 |
| `iterative` | **iterative** | the n nested loops (i₀ outermost) for a runtime rank n, implemented as an *odometer*: a counter array idx[] plus an incrementally updated source offset. There is no div/mod in the steady state, and the innermost loop is a strided copy | 1 |
| `recursive` | **recursive, 1-d data** | a rank-n reorder is d₀ independent rank-(n−1) reorders: R(d₀,…) = (I_{d₀} ⊗ R(d₁,…))·L^N_{d₀}. The stride permutation L is folded into pointer arithmetic on a flat `char*` buffer. Base case: a rank-1 strided copy | 1 |
| `recursive_nd` | **recursive, n-d data** | the same recursion on an n-d strided tensor-view type (data, rank, shape[], stride[]; indexing `t[i]` gives a rank-(r−1) view). With views, the reorder is just a copy from the column-major view of `in` to the row-major view of `out` | 1 |
| `stages` | iterative, FFT style (extra) | digit reversal as a product of n−1 stride permutations, (I ⊗ L)…(I_{d₀} ⊗ L)·L, one full pass each (ping-pong buffers) | n−1 |
| `blocked` | Part 3 | cache-blocked single pass: 64×64-byte tiles, an L1 buffer, SSE2 16×16 in-register transposes and streaming stores | 1 |

C has no arbitrary-rank array type, so `recursive_nd` defines a minimal one, in the style of a NumPy ndarray header. That gives the recursion both a 1-d and an n-d data representation. No library reorder or transpose routine is used anywhere.

How each one is parallelized (OpenMP):

* `iterative`, `index` and `stages` split the output into one contiguous, 64-byte-aligned range per thread. A thread decodes its starting position once.
* `recursive` and `recursive_nd` flatten the outermost recursion levels into one parallel loop (≥ 64 × threads iterations), and each iteration recurses.
* `blocked` makes the tiles the parallel loop.

`iterative` is the implementation used for the main plot (1…p cores).

## Bandwidth and memory traffic

The reported bandwidth is the compulsory traffic divided by time: **BW = 2N / t**, which counts N bytes loaded and N bytes stored. For N = 2³² this is 8.59 GB per reorder, whatever the rank.

Actual memory traffic depends on the algorithm and the rank. Notation: rank n, extent k = 2^{32/n}, cache line L = 64 B.

* **Ideal single pass** (any algorithm that touches every byte once): 2N. With write-allocate caches, every output line is first read (RFO), giving **3N**. Streaming (non-temporal) stores remove the RFO, giving 2N.
* **Unblocked single pass** (`index`, `iterative`, `recursive`, `recursive_nd`): stores are sequential, so they cost N + N (RFO). Loads are gathered. A 64-byte input line holds the fastest Fortran digits, which are the *slowest* C digits. Two uses of the same input line are ≥ N/k outputs apart, so the line survives only if the cache holds min(64N/k, N) bytes of lines:
  * rank 2: 4 MiB;
  * rank 4: 1 GiB;
  * rank ≥ 8: the whole 4 GiB input.

  Otherwise every byte loaded costs a whole line. So **Q(n) ≈ 64N + 2N ≈ 66N** for n ≥ 4, which is about 283 GB for 2³². Rank 2 depends on whether 4 MiB of lines at a 64 KiB stride survive set conflicts, and rank 1 is a plain copy at 3N. In practice these runs are latency bound: each line access is a cache and TLB miss that the hardware prefetcher cannot predict.
* **Index arithmetic**: `index` performs O(n) integer operations per byte (32 shift/mask/add groups at rank 32). `iterative` and `recursive` amortize this to O(1).
* **`stages`** makes n−1 passes, so its traffic is **Q(n) = (n−1)·2N** compulsory (×1.5 with RFO). This grows linearly with rank: 31 passes, or 266 GB, at rank 32. Small-radix passes (k = 2) use only half of each line per sweep, which adds more.
* **`blocked`** reads every line once completely and writes every line once completely, so **Q = 3N, or 2N with streaming stores, for every rank n ≥ 2.** This holds as long as a tile's 2 × 64 lines stay in L1, which the L1 buffer guarantees.

See `../runs/` and `../plots.pdf` for the measurements and the roofline discussion: rank 1 is a plain copy and serves as the attainable-bandwidth roof.

## Part 3: how to block rank-N digit reversal

Correctness is the same for every rank; what blocking has to fix is locality. A cache line is 64 consecutive bytes along the *fastest input digits*, but the output wants consecutive bytes along the *fastest output digits*, which are the slowest input digits. The goal is a tile that is 64 bytes wide in both directions:

1. **Group axes.** Let P be the shortest prefix of axes (0, 1, …) whose product U ≥ 64; these axes are contiguous in the input. Let Q be the shortest suffix (…, n−1) whose product V ≥ 64; these are contiguous in the output. All axes in between are "middle" axes and become plain outer loops. For rank 32 (k = 2), P is 6 axes and Q is 6 axes. For rank 2, P and Q are single axes of 65536 that get split into 64-element blocks, which is the classic blocked transpose. With u the linear index over P and v the linear index over Q:
   `out[ob + g(u) + v] = in[ib + u + h(v)]`.
   A 64×64 tile reads 64 full input lines and writes 64 full output lines. The g and h values are digit reversals over only 6 bits each, tabulated per tile.
2. **Beware power-of-two strides.** The 64 rows of a tile are 2^m bytes apart, so they all map to the *same* L1/L2 cache set and the tile thrashes. Measured: about 1 GB/s at 4 cores. Copying each input line into a contiguous 4 KiB buffer, transposing there, and writing each output line in one go fixes this: about 7 GB/s.
3. **Transpose in registers.** A scalar byte transpose then becomes the bottleneck. A 16×16 byte block is transposed with 4 rounds of `unpacklo/hi_epi8` that pair row i with row i+8. Each round rotates the 8-bit (row, column) address left by one bit, so 4 rounds swap rows and columns. This is the FFT-style stride-permutation algorithm (`stages`) again, but running inside registers, where a pass is nearly free.
4. **Avoid write-allocate.** Full 64-byte output lines are written with streaming stores (`_mm_stream_si128`), which cuts DRAM traffic from 3N to 2N.
5. **Other details:** huge pages (`madvise(MADV_HUGEPAGE)`) keep the 128 pages a tile touches in the TLB. Thread chunks are 64-byte aligned. Shapes too thin to form 64×64 tiles fall back to the iterative code.

A cache-oblivious alternative recursively halves the larger of the u and v index ranges until a block fits in cache. It gets the same 2N–3N traffic without tuning the tile size, but it still needs steps 2–4 to reach bandwidth.

The ladder can be rebuilt with `-DBLOCK_KERNEL=0..3`. `run_all.sh` measures all four kernels and logs them to `../runs/part3_kernel_ladder.txt`.
