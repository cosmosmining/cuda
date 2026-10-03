/*
 * 18-647 HW2 driver.
 *
 *   reorder test                       correctness suite (exit code != 0 on failure)
 *   reorder bench [options]            timing runs, one line per configuration
 *
 * bench options
 *   -l LIST   log2 of the element count, e.g. 32 (default) or 20,24,28,32
 *   -a LIST   algorithms: index,iterative,recursive,recursive_nd,stages,blocked
 *             or "all" (default: iterative)
 *   -t LIST   thread counts, e.g. 1,2,4 or 1-8 (default: 1..number of cores)
 *   -k LIST   ranks (default: every n dividing log2 N, i.e. every "square" shape)
 *   -r R      repetitions per configuration (default 3)
 *
 * Every bench line is followed by machine-readable lines
 *   CSV,alg,log2N,rank,extent,threads,rep,seconds,GBps
 * where GBps = 2*N bytes (N loaded + N stored) / seconds / 1e9.
 * Each configuration is checked afterwards against the definition at 2^20
 * random output positions.
 */
#define _GNU_SOURCE
#include "reorder.h"

#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/time.h>

#define MAXR REORDER_MAX_RANK

static double now(void) /* seconds, gettimeofday() as the handout suggests */
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
}

static size_t prod(int n, const size_t *d)
{
    size_t N = 1;
    for (int k = 0; k < n; k++)
        N *= d[k];
    return N;
}

static char *alloc_bytes(size_t N)
{
    const size_t align = (size_t)2 << 20; /* 2 MiB: allows huge pages */
    size_t sz = (N + align - 1) / align * align;
    char *p = aligned_alloc(align, sz ? sz : align);
    if (p == NULL) {
        fprintf(stderr, "out of memory allocating %zu bytes\n", N);
        exit(2);
    }
#ifdef MADV_HUGEPAGE
    madvise(p, sz, MADV_HUGEPAGE); /* best effort (Linux); fewer TLB misses */
#endif
    return p;
}

/* ---------------------------------------------------------------------------
 * Reference implementation, written directly from the definition
 * out[C(i)] = in[F(i)]: walk i in Fortran order (so F(i) is just x) and
 * compute C(i) by Horner's rule.  Deliberately shares no code with reorder.c.
 */
static void reference_f2c(const char *in, char *out, int n, const size_t *d)
{
    size_t i[MAXR] = {0}, N = prod(n, d);
    for (size_t x = 0; x < N; x++) {
        size_t c = 0;
        for (int k = 0; k < n; k++)
            c = c * d[k] + i[k];
        out[c] = in[x];
        for (int k = 0; k < n && ++i[k] == d[k]; k++)
            i[k] = 0;
    }
}

/* ------------------------------------------------------------------ tests */

static int n_checks, n_fail;

static void expect(int ok, const char *what)
{
    n_checks++;
    if (!ok) {
        n_fail++;
        printf("FAIL: %s\n", what);
    }
}

static void fmt_shape(char *buf, size_t len, int n, const size_t *d)
{
    int w = snprintf(buf, len, "(");
    for (int k = 0; k < n && (size_t)w < len; k++)
        w += snprintf(buf + w, len - (size_t)w, k ? ",%zu" : "%zu", d[k]);
    if ((size_t)w < len)
        snprintf(buf + w, len - (size_t)w, ")");
}

static int fails_by_alg[ALG_COUNT], checks_by_alg[ALG_COUNT];

/* Check every algorithm on one shape, with each of the given thread counts.
 * Elements are labelled by their input position, one byte ("plane") at a
 * time, so a match on every plane proves the permutation is exactly right,
 * not just right up to equal byte values.  Also checks C->F (the inverse). */
static void check_shape(int n, const size_t *d, const int *threads, int nthreads)
{
    size_t N = prod(n, d);
    char *in = alloc_bytes(N), *out = alloc_bytes(N), *ref = alloc_bytes(N);
    char *back = alloc_bytes(N), *tmp = alloc_bytes(N);
    int planes = 1, ok[ALG_COUNT];
    while (planes < 8 && (N - 1) >> (8 * planes))
        planes++;
    for (int a = 0; a < ALG_COUNT; a++)
        ok[a] = 1;
    for (int pl = 0; pl < planes; pl++) {
        for (size_t x = 0; x < N; x++)
            in[x] = (char)(x >> (8 * pl));
        reference_f2c(in, ref, n, d);
        for (int a = 0; a < ALG_COUNT; a++)
            for (int ti = 0; ti < nthreads; ti++) {
                omp_set_num_threads(threads[ti]);
                memset(out, 0x5a, N);
                ok[a] &= reorder_f2c((reorder_alg)a, in, out, n, d, tmp) == 0;
                ok[a] &= memcmp(out, ref, N) == 0;
                memset(back, 0x5a, N);
                ok[a] &= reorder_c2f((reorder_alg)a, out, back, n, d, tmp) == 0;
                ok[a] &= memcmp(back, in, N) == 0;
            }
    }
    char shape[512], what[640];
    fmt_shape(shape, sizeof shape, n, d);
    for (int a = 0; a < ALG_COUNT; a++) {
        snprintf(what, sizeof what, "%s rank %d shape %s", reorder_alg_name((reorder_alg)a),
                 n, shape);
        expect(ok[a], what);
        checks_by_alg[a]++;
        fails_by_alg[a] += !ok[a];
    }
    free(in), free(out), free(ref), free(back), free(tmp);
}

/* The example from the handout, checked with real C multidimensional arrays:
 * B holds A[K][M][N]; the result must be C[N][M][K] with C[k][j][i] == A[i][j][k]. */
static void test_handout_example(reorder_alg alg)
{
    enum { K = 6, M = 7, N = 5 }; /* K*M*N = 210 < 256: every value distinct */
    static char B[K * M * N], out[K * M * N], tmp[K * M * N];
    for (int x = 0; x < K * M * N; x++)
        B[x] = (char)x;
    char(*A)[M][N] = (char(*)[M][N])B;
    char(*C)[M][K] = (char(*)[M][K])out;
    char what[128];

    /* B is the Fortran layout of shape (N, M, K) */
    size_t dF[3] = {N, M, K};
    memset(out, 0, sizeof out);
    int ok = reorder_f2c(alg, B, out, 3, dF, tmp) == 0;
    for (int i = 0; i < K; i++)
        for (int j = 0; j < M; j++)
            for (int k = 0; k < N; k++)
                ok &= C[k][j][i] == A[i][j][k];
    snprintf(what, sizeof what, "%s handout example, f2c", reorder_alg_name(alg));
    expect(ok, what);

    /* ... equivalently the C layout of shape (K, M, N), converted to Fortran */
    size_t dC[3] = {K, M, N};
    memset(out, 0, sizeof out);
    ok = reorder_c2f(alg, B, out, 3, dC, tmp) == 0;
    for (int i = 0; i < K; i++)
        for (int j = 0; j < M; j++)
            for (int k = 0; k < N; k++)
                ok &= C[k][j][i] == A[i][j][k];
    snprintf(what, sizeof what, "%s handout example, c2f", reorder_alg_name(alg));
    expect(ok, what);
}

/* Rank 2 (matrix transpose) and rank 4, checked with native C arrays. */
static void test_native_arrays(reorder_alg alg)
{
    enum { R = 13, S = 17 }; /* 221 < 256 */
    static char a2[R * S], t2[R * S], tmp[R * S];
    for (int x = 0; x < R * S; x++)
        a2[x] = (char)x;
    char(*A)[S] = (char(*)[S])a2;
    char(*T)[R] = (char(*)[R])t2;
    size_t d2[2] = {S, R}; /* a2 = A[R][S] is the Fortran layout of (S, R) */
    int ok = reorder_f2c(alg, a2, t2, 2, d2, tmp) == 0;
    for (int i = 0; i < R; i++)
        for (int j = 0; j < S; j++)
            ok &= T[j][i] == A[i][j];
    char what[128];
    snprintf(what, sizeof what, "%s native 13x17 transpose", reorder_alg_name(alg));
    expect(ok, what);

    enum { P = 2, Q = 3, U = 4, W = 5 }; /* 120 < 256 */
    static char a4[P * Q * U * W], t4[P * Q * U * W], tmp4[P * Q * U * W];
    for (int x = 0; x < P * Q * U * W; x++)
        a4[x] = (char)x;
    char(*A4)[Q][U][W] = (char(*)[Q][U][W])a4;
    char(*T4)[U][Q][P] = (char(*)[U][Q][P])t4;
    size_t d4[4] = {W, U, Q, P};
    ok = reorder_f2c(alg, a4, t4, 4, d4, tmp4) == 0;
    for (int i = 0; i < P; i++)
        for (int j = 0; j < Q; j++)
            for (int k = 0; k < U; k++)
                for (int l = 0; l < W; l++)
                    ok &= T4[l][k][j][i] == A4[i][j][k][l];
    snprintf(what, sizeof what, "%s native 2x3x4x5 rank-4", reorder_alg_name(alg));
    expect(ok, what);
}

/* All extents 2: the reorder is the bit-reversal permutation of the FFT. */
static void test_bit_reversal(reorder_alg alg)
{
    for (int n = 1; n <= 20; n++) {
        size_t N = (size_t)1 << n, d[MAXR];
        for (int k = 0; k < n; k++)
            d[k] = 2;
        uint32_t *label = malloc(N * sizeof *label);
        char *in = alloc_bytes(N), *out = alloc_bytes(N), *tmp = alloc_bytes(N);
        int ok = 1;
        memset(label, 0, N * sizeof *label);
        for (int pl = 0; pl < 3; pl++) { /* assemble 24-bit labels bytewise */
            for (size_t x = 0; x < N; x++)
                in[x] = (char)(x >> (8 * pl));
            ok &= reorder_f2c(alg, in, out, n, d, tmp) == 0;
            for (size_t x = 0; x < N; x++)
                label[x] |= (uint32_t)(unsigned char)out[x] << (8 * pl);
        }
        for (size_t x = 0; x < N; x++) {
            size_t rev = 0;
            for (int b = 0; b < n; b++)
                rev |= ((x >> b) & 1) << (n - 1 - b);
            ok &= label[x] == rev; /* out[x] came from in[bitreverse(x)] */
        }
        char what[128];
        snprintf(what, sizeof what, "%s bit reversal 2^%d", reorder_alg_name(alg), n);
        expect(ok, what);
        free(label), free(in), free(out), free(tmp);
    }
}

static uint64_t rng_state = 0x243F6A8885A308D3ull;
static uint64_t rng(void) /* xorshift64* */
{
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return rng_state * 0x2545F4914F6CDD1Dull;
}

static int run_tests(void)
{
    const int all_threads[] = {1, 2, 3, 4, 7}, some_threads[] = {1, 4, 7};
    printf("== correctness tests ==\n");

    /* hand-checkable cases against native C arrays / bit reversal */
    for (int a = 0; a < ALG_COUNT; a++) {
        reorder_alg alg = (reorder_alg)a;
        int before = n_fail, checks = n_checks;
        omp_set_num_threads(3);
        test_handout_example(alg);
        test_native_arrays(alg);
        test_bit_reversal(alg);
        checks_by_alg[a] += n_checks - checks;
        fails_by_alg[a] += n_fail - before;
    }

    /* random shapes: small extents (incl. 1), and larger ones for ragged tiles */
    for (int trial = 0; trial < 400; trial++) {
        int n = 1 + (int)(rng() % 9);
        size_t d[MAXR], N = 1;
        size_t cap = trial % 3 == 0 ? 300 : 7;
        for (int k = 0; k < n; k++) {
            d[k] = 1 + rng() % cap;
            if (N * d[k] > ((size_t)1 << 18))
                d[k] = 1;
            N *= d[k];
        }
        check_shape(n, d, &all_threads[trial % 5], 1);
    }

    /* every "square" shape of 2^24 elements (rank n | 24) */
    for (int n = 1; n <= 24; n++) {
        if (24 % n)
            continue;
        size_t d[MAXR];
        for (int k = 0; k < n; k++)
            d[k] = (size_t)1 << (24 / n);
        check_shape(n, d, some_threads, 3);
    }
    /* non-power-of-two and non-square shapes (div/mod paths, ragged tiles) */
    size_t s1[3] = {255, 257, 253}, s2[2] = {1000, 3001}, s3[5] = {3, 5, 7, 11, 13};
    size_t s4[4] = {1, 4096, 1, 3}, s5[2] = {1 << 22, 2}, s6[3] = {2, 1 << 20, 3};
    check_shape(3, s1, all_threads, 5);
    check_shape(2, s2, all_threads, 5);
    check_shape(5, s3, some_threads, 3);
    check_shape(4, s4, some_threads, 3);
    check_shape(2, s5, some_threads, 3);
    check_shape(3, s6, some_threads, 3);

    for (int a = 0; a < ALG_COUNT; a++)
        printf("%-13s %4d checks, %d failed\n", reorder_alg_name((reorder_alg)a),
               checks_by_alg[a], fails_by_alg[a]);

    /* argument validation */
    size_t bad[2] = {4, 0};
    char buf[16];
    expect(reorder_f2c(ALG_ITERATIVE, buf, buf + 8, 2, bad, NULL) == -1, "zero extent rejected");
    expect(reorder_f2c(ALG_ITERATIVE, buf, buf + 8, 0, bad, NULL) == -1, "rank 0 rejected");
    size_t three[3] = {2, 2, 2};
    expect(reorder_f2c(ALG_STAGES, buf, buf + 8, 3, three, NULL) == -1, "stages needs tmp");

    printf("%s: %d checks, %d failed\n", n_fail ? "TESTS FAILED" : "ALL TESTS PASSED",
           n_checks, n_fail);
    return n_fail != 0;
}

/* ------------------------------------------------------------------ bench */

/* Parse "1,2,4" or "1-8" (or a mix, "1-4,8") into v[]; returns the count. */
static int parse_list(const char *s, int *v, int max)
{
    int cnt = 0;
    while (*s && cnt < max) {
        char *end;
        long a = strtol(s, &end, 10), b = a;
        if (end == s)
            break;
        if (*end == '-')
            b = strtol(end + 1, &end, 10);
        for (long x = a; x <= b && cnt < max; x++)
            v[cnt++] = (int)x;
        s = *end == ',' ? end + 1 : end;
    }
    return cnt;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* in[x]: a hash of x, so a wrong source position is caught w.p. 255/256 */
static void fill_input(char *in, size_t N)
{
#pragma omp parallel for schedule(static)
    for (size_t x = 0; x < N; x++) {
        uint64_t h = x * 0x9E3779B97F4A7C15ull;
        in[x] = (char)(h >> 56 ^ h >> 29);
    }
}

static void first_touch(char *p, size_t N)
{
#pragma omp parallel for schedule(static)
    for (size_t x = 0; x < N; x += 4096)
        memset(p + x, 0, N - x < 4096 ? N - x : 4096);
}

/* Check out[C(i)] == in[F(i)] at `samples` random output positions. */
static int verify_sampled(const char *in, const char *out, int n,
                          const size_t *d, size_t samples)
{
    size_t N = prod(n, d), sinF[MAXR];
    sinF[0] = 1;
    for (int k = 1; k < n; k++)
        sinF[k] = sinF[k - 1] * d[k - 1];
    for (size_t s = 0; s < samples; s++) {
        size_t c = s == 0 ? 0 : s == 1 ? N - 1 : rng() % N, r = c, f = 0;
        for (int k = n - 1; k >= 0; k--) { /* decode c in C order */
            f += (r % d[k]) * sinF[k];
            r /= d[k];
        }
        if (out[c] != in[f])
            return 0;
    }
    return 1;
}

static int run_bench(int argc, char **argv)
{
    int logs[8] = {32}, nlogs = 1, algs[ALG_COUNT] = {ALG_ITERATIVE}, nalgs = 1;
    int threads[256], nthreads = 0, ranks[64], nranks = 0, reps = 3;
    for (int i = 2; i < argc; i++) {
        const char *opt = argv[i], *val = i + 1 < argc ? argv[i + 1] : NULL;
        if (val == NULL) {
            fprintf(stderr, "missing value for %s\n", opt);
            return 2;
        }
        i++;
        if (!strcmp(opt, "-l")) {
            nlogs = parse_list(val, logs, 8);
        } else if (!strcmp(opt, "-t")) {
            nthreads = parse_list(val, threads, 256);
        } else if (!strcmp(opt, "-k")) {
            nranks = parse_list(val, ranks, 64);
        } else if (!strcmp(opt, "-r")) {
            reps = atoi(val);
        } else if (!strcmp(opt, "-a")) {
            nalgs = 0;
            char buf[256];
            snprintf(buf, sizeof buf, "%s", val);
            for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
                if (!strcmp(tok, "all")) {
                    for (int a = 0; a < ALG_COUNT; a++)
                        algs[nalgs++] = a;
                    break;
                }
                int a = reorder_alg_from_name(tok);
                if (a < 0 || nalgs == ALG_COUNT) {
                    fprintf(stderr, "unknown algorithm %s\n", tok);
                    return 2;
                }
                algs[nalgs++] = a;
            }
        } else {
            fprintf(stderr, "unknown option %s\n", opt);
            return 2;
        }
    }
    if (nthreads == 0)
        for (int t = 1; t <= omp_get_num_procs() && t <= 256; t++)
            threads[nthreads++] = t;
    if (reps < 1 || nlogs < 1 || nalgs < 1) {
        fprintf(stderr, "bad arguments\n");
        return 2;
    }

    int maxlog = 0, need_tmp = 0, maxthreads = 1;
    for (int i = 0; i < nlogs; i++)
        maxlog = logs[i] > maxlog ? logs[i] : maxlog;
    for (int i = 0; i < nalgs; i++)
        need_tmp |= algs[i] == ALG_STAGES;
    for (int i = 0; i < nthreads; i++)
        maxthreads = threads[i] > maxthreads ? threads[i] : maxthreads;
    if (maxlog < 1 || maxlog > 40) {
        fprintf(stderr, "log2 N must be in 1..40\n");
        return 2;
    }
    size_t Nmax = (size_t)1 << maxlog;
    printf("== bench: up to N = 2^%d = %zu one-byte elements (%.2f GiB per buffer), "
           "%d cores available, %d repetitions ==\n",
           maxlog, Nmax, (double)Nmax / (1 << 30), omp_get_num_procs(), reps);
    fflush(stdout);
    char *in = alloc_bytes(Nmax), *out = alloc_bytes(Nmax);
    char *tmp = need_tmp ? alloc_bytes(Nmax) : NULL;
    omp_set_num_threads(maxthreads);
    fill_input(in, Nmax); /* parallel first touch */
    first_touch(out, Nmax);
    if (tmp)
        first_touch(tmp, Nmax);
    size_t whole = Nmax; /* untimed warm-up: a rank-1 reorder is a plain copy */
    reorder_f2c(ALG_ITERATIVE, in, out, 1, &whole, NULL);

    for (int li = 0; li < nlogs; li++) {
        int lg = logs[li];
        size_t N = (size_t)1 << lg;
        for (int n = 1; n <= lg && n <= MAXR; n++) {
            if (lg % n)
                continue; /* only "square" shapes: extent 2^(lg/n) on every axis */
            if (nranks) {
                int want = 0;
                for (int i = 0; i < nranks; i++)
                    want |= ranks[i] == n;
                if (!want)
                    continue;
            }
            size_t d[MAXR], ext = (size_t)1 << (lg / n);
            for (int k = 0; k < n; k++)
                d[k] = ext;
            for (int ai = 0; ai < nalgs; ai++) {
                reorder_alg alg = (reorder_alg)algs[ai];
                for (int ti = 0; ti < nthreads; ti++) {
                    omp_set_num_threads(threads[ti]);
                    double t[64];
                    int R = reps < 64 ? reps : 64;
                    for (int r = 0; r < R; r++) {
                        double t0 = now();
                        if (reorder_f2c(alg, in, out, n, d, tmp) != 0) {
                            fprintf(stderr, "reorder_f2c failed\n");
                            return 1;
                        }
                        t[r] = now() - t0;
                    }
                    int ok = verify_sampled(in, out, n, d, (size_t)1 << 20);
                    double sorted[64];
                    memcpy(sorted, t, sizeof(double) * (size_t)R);
                    qsort(sorted, (size_t)R, sizeof(double), cmp_double);
                    double gb = 2.0 * (double)N / 1e9;
                    printf("%-12s N=2^%d rank=%2d extent=%-6zu threads=%3d  "
                           "time[s] min %8.4f med %8.4f max %8.4f  "
                           "BW(2N/t_min) %7.3f GB/s  check %s\n",
                           reorder_alg_name(alg), lg, n, ext, threads[ti],
                           sorted[0], sorted[R / 2], sorted[R - 1], gb / sorted[0],
                           ok ? "OK" : "FAILED");
                    for (int r = 0; r < R; r++)
                        printf("CSV,%s,%d,%d,%zu,%d,%d,%.6f,%.4f\n", reorder_alg_name(alg),
                               lg, n, ext, threads[ti], r, t[r], gb / t[r]);
                    fflush(stdout);
                    if (!ok) {
                        fprintf(stderr, "verification FAILED\n");
                        return 1;
                    }
                }
            }
        }
    }
    free(in), free(out), free(tmp);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc >= 2 && !strcmp(argv[1], "test"))
        return run_tests();
    if (argc >= 2 && !strcmp(argv[1], "bench"))
        return run_bench(argc, argv);
    fprintf(stderr,
            "usage: %s test\n"
            "       %s bench [-l 32] [-a iterative|all|a,b,..] [-t 1-4] [-k ranks] [-r 3]\n",
            argv[0], argv[0]);
    return 2;
}
