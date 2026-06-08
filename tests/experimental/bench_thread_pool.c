// bench_thread_pool.c — Compare OpenMP vs thread-pool multi-threading for GEMM
//
// Tests three dispatch strategies across BF16 and I8 SVE paths:
//   1. Original OMP  — current #pragma omp parallel for (may have 2 regions)
//   2. Fused OMP     — single #pragma omp parallel, manual work distribution
//   3. Thread pool   — persistent pthread pool with static tile partitioning
//
// Each variant is measured for:
//   - Single-call latency (cold)
//   - Repeated-call throughput (warm, amortized)
//   - Thread management overhead (% of total time)
//
// Build:
// Build:
//   cc -O3 -fopenmp -mcpu=native -o bench_thread_pool      bench_thread_pool.c bf16gemm_sve.c bf16gemm_sve.S      i8gemm_sve.c gemm_thread_pool.c -lm

#include <arm_sve.h>
#include <omp.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bf16gemm_sve.h"
#include "gemm_params.h"
#include "gemm_thread_pool.h"
#include "i8gemm.h"

// ═══════════════════════════════════════════════════════════════════════
// Timer helpers
// ═══════════════════════════════════════════════════════════════════════

static double now_sec(void) {
    struct timespec tp;
    clock_gettime(CLOCK_MONOTONIC_RAW, &tp);
    return (double)tp.tv_sec + (double)tp.tv_nsec * 1e-9;
}

// ═══════════════════════════════════════════════════════════════════════
// Utility helpers
// ═══════════════════════════════════════════════════════════════════════

static int round_up_int(int x, int q) {
    return ((x + q - 1) / q) * q;
}

static int sve_segments(void) {
    return (int)(svcntb() / 16);
}

static int sve_n_tile(void) {
    return sve_segments() * 8;
}

static void *xaligned_alloc(size_t bytes) {
    if (bytes == 0) bytes = 64;
    bytes = (bytes + 63u) & ~(size_t)63u;
    void *p = aligned_alloc(64, bytes);
    if (!p) { fprintf(stderr, "alloc failed: %zu\n", bytes); exit(1); }
    return p;
}

static bf16_t float_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    u += ((u >> 16) & 1u) + 0x7fffu;
    return (bf16_t)(u >> 16);
}

// ═══════════════════════════════════════════════════════════════════════
// Test shape definitions
// ═══════════════════════════════════════════════════════════════════════

typedef struct {
    const char *label;
    int M, K, N;
} test_shape_t;

static const test_shape_t shapes[] = {
    {"tiny",     16,   64,  128},
    {"small",    64,  256,  512},
    {"medium",  128,  512, 1024},
    {"large",   512, 1024, 4096},
    {"tall",    256,  256,  512},
    {"wide",     64,  512, 4096},
};

static const int thread_counts[] = {1, 2, 4, 8};

#define NUM_SHAPES   (int)(sizeof(shapes)   / sizeof(shapes[0]))
#define NUM_THREADS  (int)(sizeof(thread_counts) / sizeof(thread_counts[0]))
#define WARMUP_RUNS  3
#define TIMED_RUNS   10
#define REPEAT_RUNS  50   // for throughput measurement

// ═══════════════════════════════════════════════════════════════════════
// Kernel function declarations (from bf16gemm_sve.S and i8gemm_sve.c)
// ═══════════════════════════════════════════════════════════════════════

// BF16 SVE kernels
void bf16gemm_k_ld(const bf16_t *A, const bf16_t *B_reo, f32_t *C,
                   bf16_t *A_reorder, const gemm_params_t *params);
void bf16gemm_k_ld1(const bf16_t *A, const bf16_t *B_reo, f32_t *C,
                    bf16_t *A_reorder, const gemm_params_t *params);
void bf16gemm_k_ld2(const bf16_t *A, const bf16_t *B_reo, f32_t *C,
                    bf16_t *A_reorder, const gemm_params_t *params);
void bf16gemm_k_ld4(const bf16_t *A, const bf16_t *B_reo, f32_t *C,
                    bf16_t *A_reorder, const gemm_params_t *params);
void bf16gemm_k_nld_f_m12(const bf16_t *A, const bf16_t *B_reo, f32_t *C,
                          bf16_t *A_reorder, const gemm_params_t *params);

// I8 SVE kernels
void i8gemm_k_ld(const i8_t *A, const i8_t *B_reo, i32_t *C,
                 i8_t *A_reorder, const gemm_params_t *params);
void i8gemm_k_ld1(const i8_t *A, const i8_t *B_reo, i32_t *C,
                  i8_t *A_reorder, const gemm_params_t *params);
void i8gemm_k_ld2(const i8_t *A, const i8_t *B_reo, i32_t *C,
                  i8_t *A_reorder, const gemm_params_t *params);
void i8gemm_k_ld4(const i8_t *A, const i8_t *B_reo, i32_t *C,
                  i8_t *A_reorder, const gemm_params_t *params);

// ═══════════════════════════════════════════════════════════════════════
// BF16 A-reorder helpers (replicated from bf16gemm_sve.c for standalone)
// ═══════════════════════════════════════════════════════════════════════

static size_t bf16_a_reorder_stride(int K_r) {
    return (size_t)K_r * 8;
}

static void bf16_pack_A_sve_block(const bf16_t *A, bf16_t *A_reo,
                                  int M, int K_r) {
    int rowpairs = (M + 1) / 2;
    if (rowpairs < 4) rowpairs = 4;
    if (rowpairs > 6) rowpairs = 6;
    size_t idx = 0;
    for (int kb = 0; kb < K_r; kb += 4) {
        for (int rp = 0; rp < rowpairs; rp++) {
            int r0 = rp * 2;
            int r1 = r0 + 1;
            for (int k = 0; k < 4; k++)
                A_reo[idx++] = (r0 < M) ? A[(size_t)r0 * K_r + kb + k] : 0;
            for (int k = 0; k < 4; k++)
                A_reo[idx++] = (r1 < M) ? A[(size_t)r1 * K_r + kb + k] : 0;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Single-thread BF16 dispatch (for use inside parallel regions)
// ═══════════════════════════════════════════════════════════════════════

static void bf16_st_dispatch_f32(const bf16_t *A, const bf16_t *B_reo,
                                  f32_t *C, int M, int K_r, int N_r,
                                  int ldc, bf16_t *A_reo_block) {
    gemm_params_t p = {M, K_r, N_r, K_r, K_r, ldc};
    if (M <= 1)
        bf16gemm_k_ld1(A, B_reo, C, A_reo_block, &p);
    else if (M <= 2)
        bf16gemm_k_ld2(A, B_reo, C, A_reo_block, &p);
    else if (M <= 4)
        bf16gemm_k_ld4(A, B_reo, C, A_reo_block, &p);
    else
        bf16gemm_k_ld(A, B_reo, C, A_reo_block, &p);
}

// ═══════════════════════════════════════════════════════════════════════
// Single-thread I8 dispatch (for use inside parallel regions)
// ═══════════════════════════════════════════════════════════════════════

static void i8_st_dispatch_i32(const i8_t *A, const i8_t *B_reo, i32_t *C,
                                int M, int K_r, int N_r, int ldc) {
    gemm_params_t p = {M, K_r, N_r, K_r, K_r, ldc};
    if (M <= 1)
        i8gemm_k_ld1(A, B_reo, C, NULL, &p);
    else if (M <= 2)
        i8gemm_k_ld2(A, B_reo, C, NULL, &p);
    else if (M <= 4)
        i8gemm_k_ld4(A, B_reo, C, NULL, &p);
    else
        i8gemm_k_ld(A, B_reo, C, NULL, &p);
}

// ═══════════════════════════════════════════════════════════════════════
// ─── VARIANT 1: Original OpenMP dispatch (current code) ───────────────
// These call the existing bf16gemm_mt_dispatch / i8gemm_mt_dispatch which
// use #pragma omp parallel for schedule(static).  For BF16, this also
// triggers a separate parallel region inside prepare_A_reorder_pool.
// ═══════════════════════════════════════════════════════════════════════

static double bench_bf16_original(const bf16_t *A, const bf16_t *B_reo,
                                   f32_t *C, int M, int K_r, int N_r,
                                   int nth, int runs) {
    double best = 1e30;
    for (int r = 0; r < runs; r++) {
        memset(C, 0, (size_t)M * N_r * sizeof(f32_t));
        double t0 = now_sec();
        bf16gemm_mt_dispatch(A, B_reo, C, M, K_r, N_r, nth);
        double t = now_sec() - t0;
        if (t < best) best = t;
    }
    return best;
}

static double bench_i8_original(const i8_t *A, const i8_t *B_reo,
                                 i32_t *C, int M, int K_r, int N_r,
                                 int nth, int runs) {
    double best = 1e30;
    for (int r = 0; r < runs; r++) {
        memset(C, 0, (size_t)M * N_r * sizeof(i32_t));
        double t0 = now_sec();
        i8gemm_mt_dispatch(A, B_reo, C, M, K_r, N_r, nth);
        double t = now_sec() - t0;
        if (t < best) best = t;
    }
    return best;
}

// ═══════════════════════════════════════════════════════════════════════
// ─── VARIANT 2: Fused OpenMP — single #pragma omp parallel ────────────
//
// Instead of separate parallel regions for A-reorder + dispatch, we open
// ONE parallel region and manually distribute both pieces of work.
// For M-split: each thread reorders its own M-blocks then computes them.
// For N-split: we use a phased approach (A-reorder barrier + N-split).
// ═══════════════════════════════════════════════════════════════════════

static void bench_bf16_fused_dispatch(const bf16_t *A, const bf16_t *B_reo,
                                       f32_t *C, int M, int K_r, int N_r,
                                       int num_threads) {
    const int n_tile = sve_n_tile();
    const int n_tiles = N_r / n_tile;
    const int m_blocks = (M + 7) / 8;     // ceiling of M/8
    const size_t a_stride = bf16_a_reorder_stride(K_r);

    // Determine split policy
    int use_n_split = 0;
    if (num_threads > 1 && n_tiles >= num_threads && M / 8 < num_threads) {
        const size_t b_panel = (size_t)K_r * (size_t)N_r * sizeof(bf16_t);
        if (b_panel >= 512u * 1024u && N_r >= M * 2)
            use_n_split = 1;
    }

    // Pre-allocate A-reorder pool (one stride per M-block)
    bf16_t *A_reo_pool = (bf16_t *)aligned_alloc(64,
        (size_t)m_blocks * a_stride * sizeof(bf16_t));

    if (!use_n_split) {
        // ── M-split: fuse A-reorder + GEMM in one parallel region ────
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            int blocks_per = m_blocks / num_threads;
            int extra      = m_blocks % num_threads;
            int start_block, my_blocks;
            if (tid < extra) {
                start_block = tid * (blocks_per + 1);
                my_blocks   = blocks_per + 1;
            } else {
                start_block = extra * (blocks_per + 1) + (tid - extra) * blocks_per;
                my_blocks   = blocks_per;
            }

            for (int b = 0; b < my_blocks; b++) {
                int blk = start_block + b;
                int m0 = blk * 8;
                int mb = M - m0 < 8 ? M - m0 : 8;
                bf16_t *a_reo = A_reo_pool + (size_t)blk * a_stride;
                bf16_pack_A_sve_block(A + (size_t)m0 * K_r, a_reo, mb, K_r);
                bf16_st_dispatch_f32(A + (size_t)m0 * K_r, B_reo,
                                     C + (size_t)m0 * N_r,
                                     mb, K_r, N_r, N_r, a_reo);
            }
        }
    } else {
        // ── N-split: phased — shared A-reorder, then N-tile dispatch ─

        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();

            // Phase 1: M-split A-reorder (all threads participate)
            #pragma omp for schedule(static)
            for (int b = 0; b < m_blocks; b++) {
                int m0 = b * 8;
                int mb = M - m0 < 8 ? M - m0 : 8;
                bf16_pack_A_sve_block(A + (size_t)m0 * K_r,
                                      A_reo_pool + (size_t)b * a_stride,
                                      mb, K_r);
            }
            // implicit barrier here

            // Phase 2: N-split dispatch
            int tiles_per = n_tiles / num_threads;
            int extra_t   = n_tiles % num_threads;
            int start_tile, my_tiles;
            if (tid < extra_t) {
                start_tile = tid * (tiles_per + 1);
                my_tiles   = tiles_per + 1;
            } else {
                start_tile = extra_t * (tiles_per + 1) + (tid - extra_t) * tiles_per;
                my_tiles   = tiles_per;
            }

            for (int t = 0; t < my_tiles; t++) {
                int tile_idx = start_tile + t;
                int n0 = tile_idx * n_tile;
                for (int m0 = 0; m0 < M; m0 += 8) {
                    int mb = M - m0 < 8 ? M - m0 : 8;
                    bf16_t *a_reo = A_reo_pool + (size_t)(m0 / 8) * a_stride;
                    bf16_st_dispatch_f32(A + (size_t)m0 * K_r,
                                         B_reo + (size_t)tile_idx * K_r * n_tile,
                                         C + (size_t)m0 * N_r + n0,
                                         mb, K_r, n_tile, N_r, a_reo);
                }
            }
        }
    }
    free(A_reo_pool);
}

static double bench_bf16_fused(const bf16_t *A, const bf16_t *B_reo,
                                f32_t *C, int M, int K_r, int N_r,
                                int nth, int runs) {
    double best = 1e30;
    for (int r = 0; r < runs; r++) {
        memset(C, 0, (size_t)M * N_r * sizeof(f32_t));
        double t0 = now_sec();
        bench_bf16_fused_dispatch(A, B_reo, C, M, K_r, N_r, nth);
        double t = now_sec() - t0;
        if (t < best) best = t;
    }
    return best;
}

static void bench_i8_fused_dispatch(const i8_t *A, const i8_t *B_reo,
                                     i32_t *C, int M, int K_r, int N_r,
                                     int num_threads) {
    const int n_tile = sve_n_tile();
    const int n_tiles = N_r / n_tile;
    const int m_blocks = (M + 7) / 8;

    int use_n_split = 0;
    if (num_threads > 1 && n_tiles >= num_threads && M / 8 < num_threads)
        use_n_split = 1;

    if (!use_n_split) {
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            int blocks_per = m_blocks / num_threads;
            int extra      = m_blocks % num_threads;
            int start_block, my_blocks;
            if (tid < extra) {
                start_block = tid * (blocks_per + 1);
                my_blocks   = blocks_per + 1;
            } else {
                start_block = extra * (blocks_per + 1) + (tid - extra) * blocks_per;
                my_blocks   = blocks_per;
            }

            for (int b = 0; b < my_blocks; b++) {
                int m0 = (start_block + b) * 8;
                int mb = M - m0 < 8 ? M - m0 : 8;
                i8_st_dispatch_i32(A + (size_t)m0 * K_r, B_reo,
                                    C + (size_t)m0 * N_r,
                                    mb, K_r, N_r, N_r);
            }
        }
    } else {
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            int tiles_per = n_tiles / num_threads;
            int extra_t   = n_tiles % num_threads;
            int start_tile, my_tiles;
            if (tid < extra_t) {
                start_tile = tid * (tiles_per + 1);
                my_tiles   = tiles_per + 1;
            } else {
                start_tile = extra_t * (tiles_per + 1) + (tid - extra_t) * tiles_per;
                my_tiles   = tiles_per;
            }

            for (int t = 0; t < my_tiles; t++) {
                int n0 = (start_tile + t) * n_tile;
                i8_st_dispatch_i32(A, B_reo + (size_t)(start_tile + t) * K_r * n_tile,
                                    C + n0, M, K_r, n_tile, N_r);
            }
        }
    }
}

static double bench_i8_fused(const i8_t *A, const i8_t *B_reo,
                              i32_t *C, int M, int K_r, int N_r,
                              int nth, int runs) {
    double best = 1e30;
    for (int r = 0; r < runs; r++) {
        memset(C, 0, (size_t)M * N_r * sizeof(i32_t));
        double t0 = now_sec();
        bench_i8_fused_dispatch(A, B_reo, C, M, K_r, N_r, nth);
        double t = now_sec() - t0;
        if (t < best) best = t;
    }
    return best;
}

// ═══════════════════════════════════════════════════════════════════════
// ─── VARIANT 3: Thread pool dispatch ───────────────────────────────────
//
// Uses persistent pthread workers. Work is partitioned statically.
// Context struct carries all needed pointers for the kernel callback.
// ═══════════════════════════════════════════════════════════════════════

typedef struct {
    const bf16_t *A;
    const bf16_t *B_reo;
    f32_t *C;
    int M, K_r, N_r;
    int n_tile;
    bf16_t *A_reo_pool;
    size_t a_stride;
    int use_n_split;
    int m_blocks;
    int n_tiles;
} bf16_tp_ctx_t;

static void bf16_tp_kernel_m(int tile, int tid, void *ctx) {
    bf16_tp_ctx_t *c = (bf16_tp_ctx_t *)ctx;
    int m0 = tile * 8;
    int mb = c->M - m0 < 8 ? c->M - m0 : 8;
    bf16_t *a_reo = c->A_reo_pool ? c->A_reo_pool + (size_t)tile * c->a_stride : NULL;
    bf16_st_dispatch_f32(c->A + (size_t)m0 * c->K_r, c->B_reo,
                         c->C + (size_t)m0 * c->N_r,
                         mb, c->K_r, c->N_r, c->N_r, a_reo);
}

static void bf16_tp_kernel_n(int tile, int tid, void *ctx) {
    bf16_tp_ctx_t *c = (bf16_tp_ctx_t *)ctx;
    int n0 = tile * c->n_tile;
    for (int m0 = 0; m0 < c->M; m0 += 8) {
        int mb = c->M - m0 < 8 ? c->M - m0 : 8;
        bf16_t *a_reo = c->A_reo_pool ?
            c->A_reo_pool + (size_t)(m0 / 8) * c->a_stride : NULL;
        bf16_st_dispatch_f32(c->A + (size_t)m0 * c->K_r,
                             c->B_reo + (size_t)tile * c->K_r * c->n_tile,
                             c->C + (size_t)m0 * c->N_r + n0,
                             mb, c->K_r, c->n_tile, c->N_r, a_reo);
    }
}

static double bench_bf16_threadpool(gemm_thread_pool_t *tp,
                                     const bf16_t *A, const bf16_t *B_reo,
                                     f32_t *C, int M, int K_r, int N_r,
                                     int runs) {
    const int n_tile = sve_n_tile();
    const int n_tiles = N_r / n_tile;
    const int m_blocks = (M + 7) / 8;
    const size_t a_stride = bf16_a_reorder_stride(K_r);
    const int nth = gemm_tp_num_threads(tp);

    int use_n_split = 0;
    if (nth > 1 && n_tiles >= nth && M / 8 < nth) {
        const size_t b_panel = (size_t)K_r * (size_t)N_r * sizeof(bf16_t);
        if (b_panel >= 512u * 1024u && N_r >= M * 2)
            use_n_split = 1;
    }

    // Pre-allocate A reorder pool (reused across runs)
    bf16_t *A_reo_pool = NULL;
    if (!use_n_split || 1) { // always allocate for simplicity
        A_reo_pool = (bf16_t *)aligned_alloc(64,
            (size_t)m_blocks * a_stride * sizeof(bf16_t));
    }

    // Pre-reorder A for all blocks (serial or via thread pool)
    if (A_reo_pool) {
        for (int b = 0; b < m_blocks; b++) {
            int m0 = b * 8;
            int mb = M - m0 < 8 ? M - m0 : 8;
            bf16_pack_A_sve_block(A + (size_t)m0 * K_r,
                                  A_reo_pool + (size_t)b * a_stride, mb, K_r);
        }
    }

    bf16_tp_ctx_t ctx = {A, B_reo, C, M, K_r, N_r, n_tile,
                         A_reo_pool, a_stride, use_n_split, m_blocks, n_tiles};

    double best = 1e30;
    for (int r = 0; r < runs; r++) {
        memset(C, 0, (size_t)M * N_r * sizeof(f32_t));
        double t0 = now_sec();
        if (!use_n_split) {
            gemm_tp_run_tiles(tp, m_blocks, bf16_tp_kernel_m, &ctx);
        } else {
            gemm_tp_run_tiles(tp, n_tiles, bf16_tp_kernel_n, &ctx);
        }
        double t = now_sec() - t0;
        if (t < best) best = t;
    }

    free(A_reo_pool);
    return best;
}

// ── I8 thread pool ────────────────────────────────────────────────────

typedef struct {
    const i8_t *A;
    const i8_t *B_reo;
    i32_t *C;
    int M, K_r, N_r;
    int n_tile;
    int use_n_split;
    int m_blocks;
    int n_tiles;
} i8_tp_ctx_t;

static void i8_tp_kernel_m(int tile, int tid, void *ctx) {
    i8_tp_ctx_t *c = (i8_tp_ctx_t *)ctx;
    int m0 = tile * 8;
    int mb = c->M - m0 < 8 ? c->M - m0 : 8;
    i8_st_dispatch_i32(c->A + (size_t)m0 * c->K_r, c->B_reo,
                        c->C + (size_t)m0 * c->N_r,
                        mb, c->K_r, c->N_r, c->N_r);
}

static void i8_tp_kernel_n(int tile, int tid, void *ctx) {
    i8_tp_ctx_t *c = (i8_tp_ctx_t *)ctx;
    int n0 = tile * c->n_tile;
    i8_st_dispatch_i32(c->A,
                        c->B_reo + (size_t)tile * c->K_r * c->n_tile,
                        c->C + n0, c->M, c->K_r, c->n_tile, c->N_r);
}

static double bench_i8_threadpool(gemm_thread_pool_t *tp,
                                   const i8_t *A, const i8_t *B_reo,
                                   i32_t *C, int M, int K_r, int N_r,
                                   int runs) {
    const int n_tile = sve_n_tile();
    const int n_tiles = N_r / n_tile;
    const int m_blocks = (M + 7) / 8;
    const int nth = gemm_tp_num_threads(tp);

    int use_n_split = 0;
    if (nth > 1 && n_tiles >= nth && M / 8 < nth)
        use_n_split = 1;

    i8_tp_ctx_t ctx = {A, B_reo, C, M, K_r, N_r, n_tile,
                       use_n_split, m_blocks, n_tiles};

    double best = 1e30;
    for (int r = 0; r < runs; r++) {
        memset(C, 0, (size_t)M * N_r * sizeof(i32_t));
        double t0 = now_sec();
        if (!use_n_split) {
            gemm_tp_run_tiles(tp, m_blocks, i8_tp_kernel_m, &ctx);
        } else {
            gemm_tp_run_tiles(tp, n_tiles, i8_tp_kernel_n, &ctx);
        }
        double t = now_sec() - t0;
        if (t < best) best = t;
    }
    return best;
}

// ═══════════════════════════════════════════════════════════════════════
// Test data preparation
// ═══════════════════════════════════════════════════════════════════════

static void prepare_bf16_data(const bf16_t **A_out, const bf16_t **B_out,
                               bf16_t **A_pad, bf16_t **B_reo,
                               f32_t **C_out,
                               int M, int K, int N, int *K_r, int *N_r) {
    *K_r = round_up_int(K < 8 ? 8 : K, 8);
    *N_r = round_up_int(N < 8 ? 8 : N, sve_n_tile());

    bf16_t *A = (bf16_t *)xaligned_alloc((size_t)M * *K_r * sizeof(bf16_t));
    bf16_t *B = (bf16_t *)xaligned_alloc((size_t)K * N * sizeof(bf16_t));

    for (int i = 0; i < M * K; i++)
        A[i] = float_to_bf16((float)((i % 17) - 8) * 0.125f);
    for (int i = 0; i < K * N; i++)
        B[i] = float_to_bf16((float)((i % 13) - 6) * 0.125f);

    // Pad A to K_r
    if (*K_r > K) {
        for (int i = 0; i < M; i++) {
            for (int j = K; j < *K_r; j++)
                A[(size_t)i * *K_r + j] = 0;
            memmove(A + (size_t)i * *K_r, A + (size_t)i * K, (size_t)K * sizeof(bf16_t));
        }
    }
    // Re-fill after memmove
    for (int i = 0; i < M * K; i++)
        A[i] = float_to_bf16((float)((i % 17) - 8) * 0.125f);
    if (*K_r > K) {
        // Pad properly
        bf16_t *Ap = (bf16_t *)xaligned_alloc((size_t)M * *K_r * sizeof(bf16_t));
        memset(Ap, 0, (size_t)M * *K_r * sizeof(bf16_t));
        for (int i = 0; i < M; i++)
            memcpy(Ap + (size_t)i * *K_r, A + (size_t)i * K, (size_t)K * sizeof(bf16_t));
        free(A);
        A = Ap;
    } else {
        // Re-init A
        for (int i = 0; i < M * K; i++)
            A[i] = float_to_bf16((float)((i % 17) - 8) * 0.125f);
    }

    // Pack B
    bf16_t *B_pad = (bf16_t *)calloc((size_t)*K_r * *N_r, sizeof(bf16_t));
    for (int i = 0; i < K; i++)
        memcpy(B_pad + (size_t)i * *N_r, B + (size_t)i * N, (size_t)N * sizeof(bf16_t));

    *B_reo = (bf16_t *)xaligned_alloc((size_t)*K_r * *N_r * sizeof(bf16_t));
    bf16_pack_B(B_pad, *B_reo, *K_r, *N_r);
    free(B_pad);

    *A_pad = A;
    *A_out = A;
    *B_out = B;
    *C_out = (f32_t *)xaligned_alloc((size_t)M * *N_r * sizeof(f32_t));
}

static void prepare_i8_data(const i8_t **A_out, const i8_t **B_out,
                             i8_t **A_pad, i8_t **B_reo,
                             i32_t **C_out,
                             int M, int K, int N, int *K_r, int *N_r) {
    *K_r = round_up_int(K < 16 ? 16 : K, 16);
    *N_r = round_up_int(N < 8 ? 8 : N, sve_n_tile());

    i8_t *A = (i8_t *)xaligned_alloc((size_t)M * *K_r);
    i8_t *B = (i8_t *)xaligned_alloc((size_t)K * N);

    for (int i = 0; i < M * K; i++)
        A[i] = (i8_t)((i % 17) - 8);
    for (int i = 0; i < K * N; i++)
        B[i] = (i8_t)((i % 13) - 6);

    // Pad A if needed
    if (*K_r > K) {
        i8_t *Ap = (i8_t *)xaligned_alloc((size_t)M * *K_r);
        memset(Ap, 0, (size_t)M * *K_r);
        for (int i = 0; i < M; i++)
            memcpy(Ap + (size_t)i * *K_r, A + (size_t)i * K, (size_t)K);
        free(A);
        A = Ap;
    }

    // Pack B
    i8_t *B_pad = (i8_t *)calloc((size_t)*K_r * *N_r, 1);
    for (int i = 0; i < K; i++)
        memcpy(B_pad + (size_t)i * *N_r, B + (size_t)i * N, (size_t)N);
    *B_reo = (i8_t *)xaligned_alloc((size_t)*K_r * *N_r);
    i8_pack_B(B_pad, *B_reo, *K_r, *N_r);
    free(B_pad);

    *A_pad = A;
    *A_out = A;
    *B_out = B;
    *C_out = (i32_t *)xaligned_alloc((size_t)M * *N_r * sizeof(i32_t));
}

// ═══════════════════════════════════════════════════════════════════════
// Thread management overhead measurement
//
// We measure the overhead of creating/destroying thread teams by running
// a "null" parallel region (empty body) and timing it.
// ═══════════════════════════════════════════════════════════════════════

static double bench_omp_overhead(int nth, int iterations) {
    double t0 = now_sec();
    for (int i = 0; i < iterations; i++) {
        #pragma omp parallel num_threads(nth)
        {
            // empty — just team create/barrier/destroy
        }
    }
    return (now_sec() - t0) / (double)iterations;
}

static void dummy_kernel(int tile, int tid, void *ctx) {
    (void)tile; (void)tid; (void)ctx;
}

static double bench_tp_overhead(gemm_thread_pool_t *tp, int iterations) {
    // Use gemm_tp_run_tiles with 1 tile and a no-op kernel.
    // Workers wake, call dummy_kernel once, then go back to sleep.
    double t0 = now_sec();
    for (int i = 0; i < iterations; i++) {
        gemm_tp_run_tiles(tp, 1, dummy_kernel, NULL);
    }
    return (now_sec() - t0) / (double)iterations;
}

// ═══════════════════════════════════════════════════════════════════════
// Main benchmark driver
// ═══════════════════════════════════════════════════════════════════════

int main(void) {
    printf("# GEMM Multi-threading Comparison Benchmark\n");
    printf("# SVE VL=%zu bits, segments=%d, N tile=%d\n",
           svcntb() * 8, sve_segments(), sve_n_tile());
    printf("# max OMP threads=%d\n", omp_get_max_threads());
    printf("#\n");
    printf("# Variants compared:\n");
    printf("#   1. Original OMP    — current #pragma omp parallel for\n");
    printf("#   2. Fused OMP       — single #pragma omp parallel + manual distribution\n");
    printf("#   3. Thread pool     — persistent pthread workers\n");
    printf("#\n");

    // ── Thread management overhead ────────────────────────────────────
    printf("# ── Thread management overhead (empty dispatch) ──\n");
    printf("# threads,omp_us,tp_us\n");
    for (int t = 0; t < NUM_THREADS; t++) {
        int nth = thread_counts[t];
        double omp_oh = bench_omp_overhead(nth, 200) * 1e6;  // us
        gemm_thread_pool_t *tp = gemm_tp_create(nth);
        double tp_oh = bench_tp_overhead(tp, 200) * 1e6;
        printf("# %d,%.1f,%.1f\n", nth, omp_oh, tp_oh);
        gemm_tp_destroy(tp);
    }

    // ── CSV header for main results ───────────────────────────────────
    printf("\n");
    printf("dtype,shape,M,K,N,threads,variant,best_sec,gflops,throughput_gflops\n");

    for (int s = 0; s < NUM_SHAPES; s++) {
        int M = shapes[s].M, K = shapes[s].K, N = shapes[s].N;
        const char *label = shapes[s].label;

        // ── BF16 ──────────────────────────────────────────────────────
        {
            const bf16_t *A_orig, *B_orig;
            bf16_t *A_pad, *B_reo;
            f32_t *C;
            int K_r, N_r;

            prepare_bf16_data(&A_orig, &B_orig, &A_pad, &B_reo, &C,
                              M, K, N, &K_r, &N_r);

            for (int t = 0; t < NUM_THREADS; t++) {
                int nth = thread_counts[t];
                if (nth > omp_get_max_threads()) continue;

                // Warmup
                memset(C, 0, (size_t)M * N_r * sizeof(f32_t));
                bf16gemm_mt_dispatch(A_pad, B_reo, C, M, K_r, N_r, nth);

                // 1. Original OMP
                double best_omp = bench_bf16_original(A_pad, B_reo, C,
                                                       M, K_r, N_r, nth,
                                                       TIMED_RUNS);
                double gflops_omp = 2.0 * (double)M * (double)K * (double)N
                                  / best_omp * 1e-9;

                // Throughput: many repeated calls
                double t0 = now_sec();
                for (int r = 0; r < REPEAT_RUNS; r++) {
                    memset(C, 0, (size_t)M * N_r * sizeof(f32_t));
                    bf16gemm_mt_dispatch(A_pad, B_reo, C, M, K_r, N_r, nth);
                }
                double tp_omp = 2.0 * (double)M * (double)K * (double)N
                              * (double)REPEAT_RUNS / (now_sec() - t0) * 1e-9;

                printf("bf16,%s,%d,%d,%d,%d,original,%.6f,%.2f,%.2f\n",
                       label, M, K, N, nth, best_omp, gflops_omp, tp_omp);

                // 2. Fused OMP
                double best_fused = bench_bf16_fused(A_pad, B_reo, C,
                                                      M, K_r, N_r, nth,
                                                      TIMED_RUNS);
                double gflops_fused = 2.0 * (double)M * (double)K * (double)N
                                    / best_fused * 1e-9;

                t0 = now_sec();
                for (int r = 0; r < REPEAT_RUNS; r++) {
                    memset(C, 0, (size_t)M * N_r * sizeof(f32_t));
                    bench_bf16_fused_dispatch(A_pad, B_reo, C, M, K_r, N_r, nth);
                }
                double tp_fused = 2.0 * (double)M * (double)K * (double)N
                                * (double)REPEAT_RUNS / (now_sec() - t0) * 1e-9;

                printf("bf16,%s,%d,%d,%d,%d,fused_omp,%.6f,%.2f,%.2f\n",
                       label, M, K, N, nth, best_fused, gflops_fused, tp_fused);

                // 3. Thread pool
                gemm_thread_pool_t *tp_bf16 = gemm_tp_create(nth);
                // Warmup
                {
                    bf16_tp_ctx_t ctx = {A_pad, B_reo, C, M, K_r, N_r,
                                         sve_n_tile(), NULL, 0, 0,
                                         (M+7)/8, N_r/sve_n_tile()};
                    memset(C, 0, (size_t)M * N_r * sizeof(f32_t));
                    // Determine split
                    int use_n = 0;
                    if (nth > 1 && (N_r/sve_n_tile()) >= nth && M/8 < nth)
                        use_n = 1;
                    ctx.use_n_split = use_n;
                    if (!use_n)
                        gemm_tp_run_tiles(tp_bf16, ctx.m_blocks, bf16_tp_kernel_m, &ctx);
                    else
                        gemm_tp_run_tiles(tp_bf16, ctx.n_tiles, bf16_tp_kernel_n, &ctx);

                    // Pre-reorder for proper measurement
                    ctx.A_reo_pool = (bf16_t *)aligned_alloc(64,
                        (size_t)ctx.m_blocks * bf16_a_reorder_stride(K_r) * sizeof(bf16_t));
                    if (ctx.A_reo_pool) {
                        for (int b = 0; b < ctx.m_blocks; b++) {
                            int m0 = b * 8;
                            int mb = M - m0 < 8 ? M - m0 : 8;
                            bf16_pack_A_sve_block(A_pad + (size_t)m0 * K_r,
                                                  ctx.A_reo_pool + (size_t)b * bf16_a_reorder_stride(K_r),
                                                  mb, K_r);
                        }
                    }
                    ctx.a_stride = bf16_a_reorder_stride(K_r);

                    double best_tp = bench_bf16_threadpool(tp_bf16, A_pad, B_reo, C,
                                                            M, K_r, N_r, TIMED_RUNS);
                    double gflops_tp = 2.0 * (double)M * (double)K * (double)N
                                     / best_tp * 1e-9;

                    t0 = now_sec();
                    for (int r = 0; r < REPEAT_RUNS; r++) {
                        memset(C, 0, (size_t)M * N_r * sizeof(f32_t));
                        if (!ctx.use_n_split)
                            gemm_tp_run_tiles(tp_bf16, ctx.m_blocks, bf16_tp_kernel_m, &ctx);
                        else
                            gemm_tp_run_tiles(tp_bf16, ctx.n_tiles, bf16_tp_kernel_n, &ctx);
                    }
                    double tp_tp = 2.0 * (double)M * (double)K * (double)N
                                 * (double)REPEAT_RUNS / (now_sec() - t0) * 1e-9;

                    printf("bf16,%s,%d,%d,%d,%d,threadpool,%.6f,%.2f,%.2f\n",
                           label, M, K, N, nth, best_tp, gflops_tp, tp_tp);

                    free((void*)ctx.A_reo_pool);
                }
                gemm_tp_destroy(tp_bf16);
            }

            free((void*)A_pad);
            free((void*)B_orig);
            free(B_reo);
            free(C);
        }

        // ── I8 ────────────────────────────────────────────────────────
        {
            const i8_t *A_orig, *B_orig;
            i8_t *A_pad, *B_reo;
            i32_t *C;
            int K_r, N_r;

            prepare_i8_data(&A_orig, &B_orig, &A_pad, &B_reo, &C,
                            M, K, N, &K_r, &N_r);

            for (int t = 0; t < NUM_THREADS; t++) {
                int nth = thread_counts[t];
                if (nth > omp_get_max_threads()) continue;

                // Warmup
                memset(C, 0, (size_t)M * N_r * sizeof(i32_t));
                i8gemm_mt_dispatch(A_pad, B_reo, C, M, K_r, N_r, nth);

                // 1. Original OMP
                double best_omp = bench_i8_original(A_pad, B_reo, C,
                                                     M, K_r, N_r, nth,
                                                     TIMED_RUNS);
                double gops_omp = 2.0 * (double)M * (double)K * (double)N
                                / best_omp * 1e-9;

                double t0 = now_sec();
                for (int r = 0; r < REPEAT_RUNS; r++) {
                    memset(C, 0, (size_t)M * N_r * sizeof(i32_t));
                    i8gemm_mt_dispatch(A_pad, B_reo, C, M, K_r, N_r, nth);
                }
                double tp_omp = 2.0 * (double)M * (double)K * (double)N
                              * (double)REPEAT_RUNS / (now_sec() - t0) * 1e-9;

                printf("i8,%s,%d,%d,%d,%d,original,%.6f,%.2f,%.2f\n",
                       label, M, K, N, nth, best_omp, gops_omp, tp_omp);

                // 2. Fused OMP
                double best_fused = bench_i8_fused(A_pad, B_reo, C,
                                                    M, K_r, N_r, nth,
                                                    TIMED_RUNS);
                double gops_fused = 2.0 * (double)M * (double)K * (double)N
                                  / best_fused * 1e-9;

                t0 = now_sec();
                for (int r = 0; r < REPEAT_RUNS; r++) {
                    memset(C, 0, (size_t)M * N_r * sizeof(i32_t));
                    bench_i8_fused_dispatch(A_pad, B_reo, C, M, K_r, N_r, nth);
                }
                double tp_fused = 2.0 * (double)M * (double)K * (double)N
                                * (double)REPEAT_RUNS / (now_sec() - t0) * 1e-9;

                printf("i8,%s,%d,%d,%d,%d,fused_omp,%.6f,%.2f,%.2f\n",
                       label, M, K, N, nth, best_fused, gops_fused, tp_fused);

                // 3. Thread pool
                gemm_thread_pool_t *tp_i8 = gemm_tp_create(nth);
                // Warmup
                {
                    i8_tp_ctx_t ctx = {A_pad, B_reo, C, M, K_r, N_r,
                                       sve_n_tile(), 0, (M+7)/8, N_r/sve_n_tile()};
                    memset(C, 0, (size_t)M * N_r * sizeof(i32_t));
                    int use_n = 0;
                    if (nth > 1 && (N_r/sve_n_tile()) >= nth && M/8 < nth)
                        use_n = 1;
                    ctx.use_n_split = use_n;
                    if (!use_n)
                        gemm_tp_run_tiles(tp_i8, ctx.m_blocks, i8_tp_kernel_m, &ctx);
                    else
                        gemm_tp_run_tiles(tp_i8, ctx.n_tiles, i8_tp_kernel_n, &ctx);

                    double best_tp = bench_i8_threadpool(tp_i8, A_pad, B_reo, C,
                                                          M, K_r, N_r, TIMED_RUNS);
                    double gops_tp = 2.0 * (double)M * (double)K * (double)N
                                   / best_tp * 1e-9;

                    t0 = now_sec();
                    for (int r = 0; r < REPEAT_RUNS; r++) {
                        memset(C, 0, (size_t)M * N_r * sizeof(i32_t));
                        if (!ctx.use_n_split)
                            gemm_tp_run_tiles(tp_i8, ctx.m_blocks, i8_tp_kernel_m, &ctx);
                        else
                            gemm_tp_run_tiles(tp_i8, ctx.n_tiles, i8_tp_kernel_n, &ctx);
                    }
                    double tp_tp = 2.0 * (double)M * (double)K * (double)N
                                 * (double)REPEAT_RUNS / (now_sec() - t0) * 1e-9;

                    printf("i8,%s,%d,%d,%d,%d,threadpool,%.6f,%.2f,%.2f\n",
                           label, M, K, N, nth, best_tp, gops_tp, tp_tp);
                }
                gemm_tp_destroy(tp_i8);
            }

            free((void*)A_pad);
            free((void*)B_orig);
            free(B_reo);
            free(C);
        }
    }

    printf("# Done.\n");
    return 0;
}
