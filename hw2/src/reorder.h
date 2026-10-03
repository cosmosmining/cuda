/*
 * 18-647 HW2 -- rank-N tensor conversion between Fortran (column-major) and
 * C (row-major) ordering.
 *
 * Conventions used throughout (see README.md for the derivation):
 *
 *   A tensor has logical shape d[0], ..., d[n-1] and N = d[0]*...*d[n-1]
 *   one-byte (char) elements.  For a multi-index (i_0, ..., i_{n-1}):
 *
 *     Fortran offset  F(i) = i_0 + d_0*(i_1 + d_1*(i_2 + ...))   (i_0 fastest)
 *     C offset        C(i) = i_{n-1} + d_{n-1}*(i_{n-2} + ...)   (i_{n-1} fastest)
 *
 *   reorder_f2c computes   out[C(i)] = in[F(i)]   for every multi-index i.
 *   reorder_c2f computes   out[F(i)] = in[C(i)],  which is reorder_f2c with the
 *   shape reversed, so every algorithm only implements f2c.
 *
 *   Example from the handout: the buffer B holding C array A[K][M][N] is the
 *   Fortran layout of shape (N, M, K); reorder_f2c with d = {N, M, K} yields
 *   the C array C[N][M][K] with C[k][j][i] == A[i][j][k].
 *
 * All algorithms are OpenMP-parallel; the thread count is whatever OpenMP is
 * configured to use (omp_set_num_threads / OMP_NUM_THREADS).
 */
#ifndef REORDER_H
#define REORDER_H

#include <stddef.h>

#define REORDER_MAX_RANK 64

typedef enum {
    ALG_INDEX,        /* index recalculation: decode every output index        */
    ALG_ITERATIVE,    /* iterative: n nested loops emulated by an odometer     */
    ALG_RECURSIVE,    /* recursive over the rank, flat 1-d buffer + strides    */
    ALG_RECURSIVE_ND, /* same recursion on an n-d strided tensor-view type     */
    ALG_STAGES,       /* iterative, FFT style: n-1 stride-permutation passes   */
    ALG_BLOCKED,      /* Part 3: cache-blocked single pass                     */
    ALG_COUNT
} reorder_alg;

/* Name used on the command line / in logs, and the reverse lookup (-1 if unknown). */
const char *reorder_alg_name(reorder_alg alg);
int reorder_alg_from_name(const char *name);

/* Does this algorithm need the scratch buffer `tmp` (N bytes)?  Only ALG_STAGES
 * with rank >= 3 does; everyone else ignores `tmp` (may be NULL). */
int reorder_needs_tmp(reorder_alg alg, int n);

/* Fortran order -> C order.  `in` and `out` must not overlap.
 * Returns 0 on success, -1 on bad arguments (rank, zero extent, overflow,
 * missing tmp). */
int reorder_f2c(reorder_alg alg, const char *in, char *out, int n,
                const size_t *dims, char *tmp);

/* C order -> Fortran order (the inverse of reorder_f2c for the same dims). */
int reorder_c2f(reorder_alg alg, const char *in, char *out, int n,
                const size_t *dims, char *tmp);

#endif
