/*
 * 18-647 HW2 -- rank-N tensor reordering, Fortran (column-major) <-> C
 * (row-major).  See reorder.h for the conventions and README.md for the
 * design discussion and the memory-traffic derivation.
 *
 * Every algorithm computes out[C(i)] = in[F(i)] for all multi-indices i:
 *
 *   input  (Fortran) strides  sin[k]  = d_0 * ... * d_{k-1}      (sin[0]    = 1)
 *   output (C)       strides  sout[k] = d_{k+1} * ... * d_{n-1}  (sout[n-1] = 1)
 *
 * The output is always produced in order (sequential stores); the input is
 * gathered with the Fortran strides.
 *
 * Sources: the Kronecker-product formulas for stride permutations and digit
 * reversal follow J. Johnson et al., "A methodology for designing, modifying,
 * and implementing Fourier transform algorithms on various architectures"
 * (1990) and the SPIRAL notation used in 18-647 lectures.
 */
#include "reorder.h"

#include <omp.h>
#include <stdint.h>

typedef struct {
    int n;                            /* rank                                 */
    size_t N;                         /* number of elements                   */
    size_t d[REORDER_MAX_RANK];       /* extents                              */
    size_t sin[REORDER_MAX_RANK];     /* input (Fortran) strides              */
    size_t sout[REORDER_MAX_RANK];    /* output (C) strides                   */
    int pow2;                         /* all extents powers of two?           */
    unsigned ish[REORDER_MAX_RANK];   /* log2(sin[k])  (valid if pow2)        */
    unsigned osh[REORDER_MAX_RANK];   /* log2(sout[k]) (valid if pow2)        */
} plan_t;

static size_t min_sz(size_t a, size_t b) { return a < b ? a : b; }

static int plan_init(plan_t *p, int n, const size_t *d)
{
    if (n < 1 || n > REORDER_MAX_RANK || d == NULL)
        return -1;
    p->n = n;
    p->N = 1;
    p->pow2 = 1;
    for (int k = 0; k < n; k++) {
        if (d[k] == 0 || p->N > SIZE_MAX / d[k])
            return -1;
        p->d[k] = d[k];
        p->N *= d[k];
        if (d[k] & (d[k] - 1))
            p->pow2 = 0;
    }
    p->sin[0] = 1;
    for (int k = 1; k < n; k++)
        p->sin[k] = p->sin[k - 1] * d[k - 1];
    p->sout[n - 1] = 1;
    for (int k = n - 2; k >= 0; k--)
        p->sout[k] = p->sout[k + 1] * d[k + 1];
    if (p->pow2)
        for (int k = 0; k < n; k++) {
            p->ish[k] = (unsigned)__builtin_ctzll(p->sin[k]);
            p->osh[k] = (unsigned)__builtin_ctzll(p->sout[k]);
        }
    return 0;
}

/* dst[t] = src[t * stride], t < cnt.  The unit-stride case is split off so
 * the compiler vectorizes it (it is the rank-1 identity). */
static inline void strided_copy(char *restrict dst, const char *restrict src,
                                size_t cnt, size_t stride)
{
    if (stride == 1)
        for (size_t t = 0; t < cnt; t++)
            dst[t] = src[t];
    else
        for (size_t t = 0; t < cnt; t++)
            dst[t] = src[t * stride];
}

/* The calling thread's contiguous share [*lo, *hi) of N output elements.
 * Boundaries are multiples of 64 so no two threads store to one cache line. */
static void my_range(size_t N, size_t *lo, size_t *hi)
{
    size_t T = (size_t)omp_get_num_threads(), t = (size_t)omp_get_thread_num();
    size_t chunk = ((N + T - 1) / T + 63) / 64 * 64;
    *lo = min_sz(N, t * chunk);
    *hi = min_sz(N, *lo + chunk);
}

typedef void range_fn(const plan_t *p, const char *in, char *out,
                      size_t lo, size_t hi);

/* Parallelization for the range-based algorithms: split the output. */
static void run_ranges(range_fn *f, const plan_t *p, const char *in, char *out)
{
#pragma omp parallel
    {
        size_t lo, hi;
        my_range(p->N, &lo, &hi);
        if (lo < hi)
            f(p, in, out, lo, hi);
    }
}

/* ---- 1. Index recalculation -----------------------------------------------
 * Every output position j is decoded into its multi-index (C order, last
 * digit fastest) and re-encoded with the Fortran strides.  O(n) integer ops
 * per element, but all elements are independent.  With power-of-two extents
 * the digits are bit fields, so div/mod become shift/mask; with all d_k = 2
 * this is literally the FFT bit-reversal index computation. */
static void index_range(const plan_t *p, const char *in, char *out,
                        size_t lo, size_t hi)
{
    const int n = p->n;
    if (p->pow2) {
        for (size_t j = lo; j < hi; j++) {
            size_t src = 0;
            for (int k = 0; k < n; k++)
                src += ((j >> p->osh[k]) & (p->d[k] - 1)) << p->ish[k];
            out[j] = in[src];
        }
    } else {
        for (size_t j = lo; j < hi; j++) {
            size_t src = 0, r = j;
            for (int k = n - 1; k >= 0; k--) {
                src += (r % p->d[k]) * p->sin[k];
                r /= p->d[k];
            }
            out[j] = in[src];
        }
    }
}

/* ---- 2. Iterative -----------------------------------------------------------
 * The natural code is n nested loops (i_0 outermost ... i_{n-1} innermost,
 * i.e. output order).  For a run-time rank the loop counters live in idx[]
 * and are advanced like an odometer; the source offset follows incrementally
 * (+sin[k] per step, -d[k]*sin[k] on wrap-around), so there is no div/mod in
 * the steady state.  The innermost loop is a strided copy.  A thread starting
 * at output position lo decodes lo once. */
static void iterative_range(const plan_t *p, const char *in, char *out,
                            size_t lo, size_t hi)
{
    const int n = p->n;
    size_t idx[REORDER_MAX_RANK], src = 0, r = lo;
    for (int k = n - 1; k >= 0; k--) {
        idx[k] = r % p->d[k];
        r /= p->d[k];
        src += idx[k] * p->sin[k];
    }
    const size_t dl = p->d[n - 1], sl = p->sin[n - 1];
    for (size_t j = lo; j < hi;) {
        /* innermost loop: the rest of the current row */
        size_t run = min_sz(dl - idx[n - 1], hi - j);
        strided_copy(out + j, in + src, run, sl);
        j += run;
        /* row done: reset the innermost counter and carry into the outer ones */
        src -= idx[n - 1] * sl;
        idx[n - 1] = 0;
        for (int k = n - 2; k >= 0; k--) {
            src += p->sin[k];
            if (++idx[k] < p->d[k])
                break;
            src -= p->d[k] * p->sin[k];
            idx[k] = 0;
        }
    }
}

/* ---- 3. Recursive, flat 1-d buffer ------------------------------------------
 * A rank-n reorder is d_0 independent rank-(n-1) reorders: the output slab
 * out[i_0][...] (contiguous) is the reorder of the input sub-tensor
 * in(i_0, ...) (elements d_0 apart).  In Kronecker-product notation
 *
 *     R(d_0, ..., d_{n-1}) = (I_{d_0} (x) R(d_1, ..., d_{n-1})) . L^N_{d_0}
 *
 * where the stride permutation L^N_{d_0} is folded into the address
 * arithmetic, so the recursion makes a single pass over the data.
 * rec() reorders the slice that starts at axis k; a rank-1 slice is a
 * strided copy. */
static void rec(const plan_t *p, int k, const char *in, char *out)
{
    if (k == p->n - 1) {
        strided_copy(out, in, p->d[k], p->sin[k]);
        return;
    }
    for (size_t i = 0; i < p->d[k]; i++)
        rec(p, k + 1, in + i * p->sin[k], out + i * p->sout[k]);
}

/* ---- 4. Recursive, n-d tensor type ------------------------------------------
 * C has no arbitrary-rank array type, so a minimal strided view is defined
 * (like a NumPy ndarray header): data pointer, rank, shape[], stride[].
 * Indexing the first axis, t[i], gives a rank-(r-1) view that shares the
 * shape/stride arrays.  With views the reorder is nothing but a copy from the
 * column-major view of `in` to the row-major view of `out`, and that copy
 * recurses on the rank exactly like rec() above. */
typedef struct {
    char *data;
    int rank;
    const size_t *shape;
    const size_t *stride; /* in elements */
} tensor_t;

static tensor_t tensor_index(tensor_t t, size_t i) /* t[i] */
{
    return (tensor_t){t.data + i * t.stride[0], t.rank - 1, t.shape + 1,
                      t.stride + 1};
}

static void tensor_copy(tensor_t dst, tensor_t src) /* dst[...] = src[...] */
{
    if (dst.rank == 0) {
        *dst.data = *src.data;
        return;
    }
    if (dst.rank == 1) { /* the rank-0 recursion, unrolled one level */
        for (size_t i = 0; i < dst.shape[0]; i++)
            dst.data[i * dst.stride[0]] = src.data[i * src.stride[0]];
        return;
    }
    for (size_t i = 0; i < dst.shape[0]; i++)
        tensor_copy(tensor_index(dst, i), tensor_index(src, i));
}

/* Parallelization shared by both recursive variants: the outermost L levels
 * of the recursion (L < n) are flattened into one parallel loop with enough
 * iterations to balance the threads; every iteration recurses from level L.
 * A rank-1 tensor is its own reorder and is simply copied in parallel. */
static void recursive_parallel(const plan_t *p, const char *in, char *out, int nd)
{
    const int n = p->n;
    if (n == 1) {
        run_ranges(iterative_range, p, in, out);
        return;
    }
    size_t outer = 1, want = 64 * (size_t)omp_get_max_threads();
    int L = 0;
    while (L < n - 1 && outer < want)
        outer *= p->d[L++];
    /* the n-d views of the whole input and output */
    tensor_t src = {(char *)in, n, p->d, p->sin}; /* read only */
    tensor_t dst = {out, n, p->d, p->sout};

#pragma omp parallel for schedule(static)
    for (size_t t = 0; t < outer; t++) {
        /* multi-index (i_0, ..., i_{L-1}) of iteration t, C order */
        size_t r = t, io = 0, oo = 0;
        tensor_t s = src, o = dst;
        size_t ii[REORDER_MAX_RANK];
        for (int k = L - 1; k > 0; k--) {
            ii[k] = r % p->d[k];
            r /= p->d[k];
        }
        ii[0] = r;
        if (nd) {
            for (int k = 0; k < L; k++) {
                s = tensor_index(s, ii[k]);
                o = tensor_index(o, ii[k]);
            }
            tensor_copy(o, s);
        } else {
            for (int k = 0; k < L; k++) {
                io += ii[k] * p->sin[k];
                oo += ii[k] * p->sout[k];
            }
            rec(p, L, in + io, out + oo);
        }
    }
}

/* ---- 5. Iterative, FFT style: n-1 stride-permutation passes -----------------
 * Digit reversal factors into stride permutations, as in iterative FFTs:
 *
 *   R = (I_{d_0...d_{n-3}} (x) L_{d_{n-2}}) ... (I_{d_0} (x) L^{N/d_0}_{d_1}) . L^N_{d_0}
 *
 * Pass s moves axis s from the fastest to the slowest position inside every
 * block of S = N/(d_0...d_{s-1}) elements, i.e. transposes an (S/d_s x d_s)
 * matrix into a (d_s x S/d_s) one.  Buffers ping-pong between out and tmp so
 * that the last pass lands in out.  Clean, but it streams the whole tensor
 * n-1 times instead of once. */
static void stage_range(const char *x, char *y, size_t S, size_t d,
                        size_t lo, size_t hi)
{
    /* y[b*S + i*R + r] = x[b*S + r*d + i],  i < d, r < R = S/d */
    const size_t R = S / d;
    size_t b = lo / S, i = lo % S / R, r = lo % R;
    for (size_t j = lo; j < hi;) {
        size_t run = min_sz(R - r, hi - j);
        strided_copy(y + j, x + b * S + r * d + i, run, d);
        j += run;
        r = 0;
        if (++i == d) {
            i = 0;
            b++;
        }
    }
}

static void stages(const plan_t *p, const char *in, char *out, char *tmp)
{
    const int passes = p->n - 1;
    if (passes == 0) {
        run_ranges(iterative_range, p, in, out);
        return;
    }
    const char *x = in;
    size_t S = p->N;
    for (int s = 0; s < passes; s++) {
        char *y = (passes - 1 - s) % 2 == 0 ? out : tmp;
#pragma omp parallel
        {
            size_t lo, hi;
            my_range(p->N, &lo, &hi);
            if (lo < hi)
                stage_range(x, y, S, p->d[s], lo, hi);
        }
        x = y;
        S /= p->d[s];
    }
}

/* ---- 6. Part 3: cache-blocked single pass -----------------------------------
 * The fastest input axes 0..pa-1 (product U >= TILE) are contiguous in the
 * input; the fastest output axes qa..n-1 (product V >= TILE) are contiguous
 * in the output.  A tile fixes the middle axes pa..qa-1 and a TILE x TILE
 * block of (u, v), where u is the linear index over the input-fast axes and
 * v the linear index over the output-fast axes:
 *
 *     out[ob + g(u) + v] = in[ib + u + h(v)]
 *
 * So a tile reads TILE input rows of TILE contiguous bytes and writes TILE
 * output rows of TILE contiguous bytes: whole cache lines only, and the
 * tile's 2*TILE lines (8 KiB) fit in L1.  g(u) and h(v) (the digit reversal
 * restricted to the fast axes) are tabulated per tile.  Tensors too small or
 * too thin to form such tiles use the iterative code.
 *
 * Tile kernels, from the Part 3 exploration (select with -DBLOCK_KERNEL=k):
 *   0  scalar, straight from the input rows to the output rows.  The rows are
 *      2^m bytes apart, so all of them fall into the same L1/L2 cache set and
 *      the tile does NOT stay in cache (conflict misses).
 *   1  scalar, through a contiguous 4 KiB buffer: every input line is read
 *      in one go and every output line written in one go.
 *   2  as 1, full tiles transposed with SSE2 16x16 in-register transposes.
 *   3  as 2, plus non-temporal (streaming) stores for the output lines, which
 *      skip the read-for-ownership of the destination (3N -> 2N bytes).
 */
#ifndef TILE
#define TILE 64 /* one cache line; must be a multiple of 16 */
#endif
#ifndef BLOCK_KERNEL
#define BLOCK_KERNEL 3
#endif
#if BLOCK_KERNEL >= 2 && defined(__SSE2__)
#include <emmintrin.h>
#define SIMD_TILES 1
#else
#define SIMD_TILES 0
#endif

/* res[j] = sum_a digit_a(x0 + j) * stride[k_a] for j < cnt, where the
 * mixed-radix number has `na` digits on axes k_a = first + a*step (a = 0 is
 * the fastest digit). */
static void digit_offsets(const plan_t *p, int first, int step, int na,
                          const size_t *stride, size_t x0, size_t cnt,
                          size_t *res)
{
    size_t dig[REORDER_MAX_RANK], off = 0;
    for (int a = 0, k = first; a < na; a++, k += step) {
        dig[a] = x0 % p->d[k];
        x0 /= p->d[k];
        off += dig[a] * stride[k];
    }
    for (size_t j = 0; j < cnt; j++) {
        res[j] = off;
        for (int a = 0, k = first; a < na; a++, k += step) {
            off += stride[k];
            if (++dig[a] < p->d[k])
                break;
            off -= p->d[k] * stride[k];
            dig[a] = 0;
        }
    }
}

#if SIMD_TILES
/* dst row j = column j of the 16x16 byte block src.  Each round pairs row i
 * with row i+8 and interleaves their bytes, which rotates the 8-bit address
 * (row bits | column bits) left by one; four rounds swap rows and columns. */
static inline void transpose16(const char *src, size_t ss, char *dst, size_t ds)
{
    __m128i x[16], y[16];
    for (int i = 0; i < 16; i++)
        x[i] = _mm_loadu_si128((const __m128i *)(src + (size_t)i * ss));
    for (int round = 0; round < 4; round++) {
        for (int i = 0; i < 8; i++) {
            y[2 * i] = _mm_unpacklo_epi8(x[i], x[i + 8]);
            y[2 * i + 1] = _mm_unpackhi_epi8(x[i], x[i + 8]);
        }
        for (int i = 0; i < 16; i++)
            x[i] = y[i];
    }
    for (int i = 0; i < 16; i++)
        _mm_storeu_si128((__m128i *)(dst + (size_t)i * ds), x[i]);
}
#endif

/* One tile: dst[a][b] = src[b][a] for a < bu, b < bv (src rows are input
 * lines, dst rows are output lines).  `nt`: dst rows are 16-byte aligned, so
 * streaming stores may be used. */
static void tile_copy(const char *const *src, char *const *dst, size_t bu,
                      size_t bv, int nt)
{
#if BLOCK_KERNEL == 0
    (void)nt;
    for (size_t b = 0; b < bv; b++)
        for (size_t a = 0; a < bu; a++)
            dst[a][b] = src[b][a];
#else
    _Alignas(64) char buf[TILE * TILE];
    for (size_t b = 0; b < bv; b++) /* read each input line in one go */
        for (size_t a = 0; a < bu; a++)
            buf[b * TILE + a] = src[b][a];
#if SIMD_TILES
    if (bu == TILE && bv == TILE) {
        _Alignas(64) char tr[TILE * TILE];
        for (size_t b0 = 0; b0 < TILE; b0 += 16)
            for (size_t a0 = 0; a0 < TILE; a0 += 16)
                transpose16(buf + b0 * TILE + a0, TILE, tr + a0 * TILE + b0, TILE);
        for (size_t a = 0; a < TILE; a++) /* write each output line in one go */
            for (size_t c = 0; c < TILE; c += 16) {
                __m128i v = _mm_load_si128((const __m128i *)(tr + a * TILE + c));
#if BLOCK_KERNEL >= 3
                if (nt) {
                    _mm_stream_si128((__m128i *)(dst[a] + c), v);
                    continue;
                }
#endif
                _mm_storeu_si128((__m128i *)(dst[a] + c), v);
            }
        return;
    }
#endif
    (void)nt;
    for (size_t a = 0; a < bu; a++) /* write each output line in one go */
        for (size_t b = 0; b < bv; b++)
            dst[a][b] = buf[b * TILE + a];
#endif
}

static void blocked(const plan_t *p, const char *in, char *out)
{
    const int n = p->n;
    int pa = 0, qa = n;
    size_t U = 1, V = 1;
    while (pa < n && U < TILE)
        U *= p->d[pa++];
    while (qa > pa && V < TILE)
        V *= p->d[--qa];
    if (U < TILE || V < TILE) {
        run_ranges(iterative_range, p, in, out);
        return;
    }
    const size_t nu = (U + TILE - 1) / TILE, nv = (V + TILE - 1) / TILE;
    const size_t tiles = p->N / U / V * nv * nu;
    /* output rows start at out + (multiple of V) + (multiple of TILE) */
    const int nt = (uintptr_t)out % 16 == 0 && V % 16 == 0;

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (size_t t = 0; t < tiles; t++) {
            size_t u0 = t % nu * TILE, v0 = t / nu % nv * TILE, m = t / nu / nv;
            size_t bu = min_sz(TILE, U - u0), bv = min_sz(TILE, V - v0);
            size_t ib = 0, ob = 0, g[TILE], h[TILE];
            const char *src[TILE];
            char *dst[TILE];
            for (int k = qa - 1; k >= pa; k--) { /* middle axes, C order */
                size_t i = m % p->d[k];
                m /= p->d[k];
                ib += i * p->sin[k];
                ob += i * p->sout[k];
            }
            digit_offsets(p, 0, +1, pa, p->sout, u0, bu, g);       /* u -> out */
            digit_offsets(p, n - 1, -1, n - qa, p->sin, v0, bv, h); /* v -> in  */
            for (size_t b = 0; b < bv; b++)
                src[b] = in + ib + u0 + h[b];
            for (size_t a = 0; a < bu; a++)
                dst[a] = out + ob + v0 + g[a];
            tile_copy(src, dst, bu, bv, nt);
        }
#if SIMD_TILES && BLOCK_KERNEL >= 3
        _mm_sfence(); /* make this thread's streaming stores globally visible */
#endif
    }
}

/* ---- public interface -------------------------------------------------------- */

static const char *const names[ALG_COUNT] = {
    [ALG_INDEX] = "index",         [ALG_ITERATIVE] = "iterative",
    [ALG_RECURSIVE] = "recursive", [ALG_RECURSIVE_ND] = "recursive_nd",
    [ALG_STAGES] = "stages",       [ALG_BLOCKED] = "blocked",
};

const char *reorder_alg_name(reorder_alg alg)
{
    return (unsigned)alg < ALG_COUNT ? names[alg] : "?";
}

int reorder_alg_from_name(const char *name)
{
    for (int a = 0; a < ALG_COUNT; a++) {
        const char *s = names[a], *t = name;
        while (*s && *s == *t)
            s++, t++;
        if (*s == 0 && *t == 0)
            return a;
    }
    return -1;
}

int reorder_needs_tmp(reorder_alg alg, int n)
{
    return alg == ALG_STAGES && n >= 3;
}

int reorder_f2c(reorder_alg alg, const char *in, char *out, int n,
                const size_t *dims, char *tmp)
{
    plan_t p;
    if (plan_init(&p, n, dims) != 0 || in == NULL || out == NULL)
        return -1;
    if (reorder_needs_tmp(alg, n) && tmp == NULL)
        return -1;
    switch (alg) {
    case ALG_INDEX:        run_ranges(index_range, &p, in, out); break;
    case ALG_ITERATIVE:    run_ranges(iterative_range, &p, in, out); break;
    case ALG_RECURSIVE:    recursive_parallel(&p, in, out, 0); break;
    case ALG_RECURSIVE_ND: recursive_parallel(&p, in, out, 1); break;
    case ALG_STAGES:       stages(&p, in, out, tmp); break;
    case ALG_BLOCKED:      blocked(&p, in, out); break;
    default:               return -1;
    }
    return 0;
}

int reorder_c2f(reorder_alg alg, const char *in, char *out, int n,
                const size_t *dims, char *tmp)
{
    /* C order of shape d == Fortran order of the reversed shape (with the
     * multi-index reversed), so C->F for d is F->C for reverse(d). */
    size_t rd[REORDER_MAX_RANK];
    if (n < 1 || n > REORDER_MAX_RANK || dims == NULL)
        return -1;
    for (int k = 0; k < n; k++)
        rd[k] = dims[n - 1 - k];
    return reorder_f2c(alg, in, out, n, rd, tmp);
}
