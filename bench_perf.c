// bench_perf.c
// Unified benchmark: peak instruction throughput (bfmmla/smmla) +
// shape.csv single-thread test + multi-threaded BF16/I8 GEMM test.
//
// Build:
//   cc -o bench_perf bench_perf.c bf16gemm_mt.c i8gemm_mt.c
//      i8gemm_k.S i8gemm_k_bias.S bf16gemm_k.S bf16gemm_k_bias.S
//      -march=armv8.6-a+bf16+i8mm -O2 -Wall -fopenmp -lm
//
// Usage:
//   ./bench_perf [shape.csv]                    single-thread shape bench (default)
//   ./bench_perf --mt M K N [nthreads]          multi-threaded BF16 GEMM bench
//   ./bench_perf --mt-i8 M K N [nthreads]       multi-threaded I8 GEMM bench
//   ./bench_perf --mt-both M K N [nthreads]     multi-threaded BF16+I8 GEMM bench
//   ./bench_perf --mt-sweep M K N               thread sweep: BF16 (1,2,4,8,10,16,20,32,40,64)
//   ./bench_perf --mt-sweep-i8 M K N            thread sweep: I8
//   ./bench_perf --mt-sweep-both M K N          thread sweep: BF16+I8
//
// Sweep mode: tests nthreads < ncores only (no oversubscription).
//   Set OMP_PLACES=cores OMP_PROC_BIND=close for core affinity.
//   For best results, wrap with taskset: taskset -c 0-$(nproc-1) ./bench_perf --mt-sweep M K N
//
// Output (sorted by efficiency descending):
//   M K N  bf16_eff%  bf16_GFLOPS  i8_eff%  i8_GOPS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

// ═══════════════════════════════════════════════════════════════════════
// Timing
// ═══════════════════════════════════════════════════════════════════════
static double get_time(struct timespec *s, struct timespec *e) {
    return (e->tv_sec - s->tv_sec) + (e->tv_nsec - s->tv_nsec) * 1e-9;
}

// ═══════════════════════════════════════════════════════════════════════
// Peak instruction throughput — bfmmla
// 16 independent bfmmla per iteration, 16 flops each = 256 flops/iter
// ═══════════════════════════════════════════════════════════════════════
static double measure_peak_bfmmla(void) {
    volatile int64_t n;
    struct timespec s, e;

    // 16 independent bfmmla per iteration, 16 flops each = 256 flops/iter
    // Use 16 different input register pairs to distribute read-port pressure
    #define BFM_LOOP_BODY                                          \
        "bfmmla v0.4s,  v16.8h, v24.8h\n\t"                       \
        "bfmmla v1.4s,  v17.8h, v25.8h\n\t"                       \
        "bfmmla v2.4s,  v18.8h, v26.8h\n\t"                       \
        "bfmmla v3.4s,  v19.8h, v27.8h\n\t"                       \
        "bfmmla v4.4s,  v20.8h, v28.8h\n\t"                       \
        "bfmmla v5.4s,  v21.8h, v29.8h\n\t"                       \
        "bfmmla v6.4s,  v22.8h, v30.8h\n\t"                       \
        "bfmmla v7.4s,  v23.8h, v31.8h\n\t"                       \
        "bfmmla v8.4s,  v16.8h, v24.8h\n\t"                       \
        "bfmmla v9.4s,  v17.8h, v25.8h\n\t"                       \
        "bfmmla v10.4s, v18.8h, v26.8h\n\t"                       \
        "bfmmla v11.4s, v19.8h, v27.8h\n\t"                       \
        "bfmmla v12.4s, v20.8h, v28.8h\n\t"                       \
        "bfmmla v13.4s, v21.8h, v29.8h\n\t"                       \
        "bfmmla v14.4s, v22.8h, v30.8h\n\t"                       \
        "bfmmla v15.4s, v23.8h, v31.8h"

    // ── Warmup + Best of 7 runs, fixed 30M iterations ──
    int64_t iters = 30000000LL;  // 30M * 256 flops = 7.68 Gflops, ~0.07s at 100 GFLOPS
    double best_gflops = 0.0;
    for (int run = 0; run < 7; run++) {
        n = iters;
        clock_gettime(CLOCK_MONOTONIC_RAW, &s);
        __asm__ volatile (
            "movi v16.8h, #0x3c, lsl #8\n\t"
            "movi v17.8h, #0x3c, lsl #8\n\t"
            "movi v18.8h, #0x3c, lsl #8\n\t"
            "movi v19.8h, #0x3c, lsl #8\n\t"
            "movi v20.8h, #0x3c, lsl #8\n\t"
            "movi v21.8h, #0x3c, lsl #8\n\t"
            "movi v22.8h, #0x3c, lsl #8\n\t"
            "movi v23.8h, #0x3c, lsl #8\n\t"
            "movi v24.8h, #0x3c, lsl #8\n\t"
            "movi v25.8h, #0x3c, lsl #8\n\t"
            "movi v26.8h, #0x3c, lsl #8\n\t"
            "movi v27.8h, #0x3c, lsl #8\n\t"
            "movi v28.8h, #0x3c, lsl #8\n\t"
            "movi v29.8h, #0x3c, lsl #8\n\t"
            "movi v30.8h, #0x3c, lsl #8\n\t"
            "movi v31.8h, #0x3c, lsl #8\n\t"
            "movi v0.4s, #0\n\t"
            "movi v1.4s, #0\n\t"
            "movi v2.4s, #0\n\t"
            "movi v3.4s, #0\n\t"
            "movi v4.4s, #0\n\t"
            "movi v5.4s, #0\n\t"
            "movi v6.4s, #0\n\t"
            "movi v7.4s, #0\n\t"
            "movi v8.4s, #0\n\t"
            "movi v9.4s, #0\n\t"
            "movi v10.4s, #0\n\t"
            "movi v11.4s, #0\n\t"
            "movi v12.4s, #0\n\t"
            "movi v13.4s, #0\n\t"
            "movi v14.4s, #0\n\t"
            "movi v15.4s, #0\n\t"
            "1:\n\t"
            BFM_LOOP_BODY "\n\t"
            "subs %[n], %[n], #1\n\t"
            "b.ne 1b\n\t"
            : [n] "+r" (n)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
              "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
              "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
              "cc", "memory"
        );
        clock_gettime(CLOCK_MONOTONIC_RAW, &e);
        double elapsed = get_time(&s, &e);
        if (elapsed > 0.0) {
            double gflops = iters * 256.0 / elapsed * 1e-9;
            if (gflops > best_gflops) best_gflops = gflops;
        }
    }
    #undef BFM_LOOP_BODY
    return best_gflops;
}

// ═══════════════════════════════════════════════════════════════════════
// Peak instruction throughput — smmla
// 16 independent smmla per iteration, 64 ops each = 1024 ops/iter
// ═══════════════════════════════════════════════════════════════════════
static double measure_peak_smmla(void) {
    volatile int64_t n;
    struct timespec s, e;

    // 16 independent smmla per iteration, 64 ops each = 1024 ops/iter
    // Use 16 different input register pairs to distribute read-port pressure
    #define SMM_LOOP_BODY                                          \
        "smmla v0.4s,  v16.16b, v24.16b\n\t"                      \
        "smmla v1.4s,  v17.16b, v25.16b\n\t"                      \
        "smmla v2.4s,  v18.16b, v26.16b\n\t"                      \
        "smmla v3.4s,  v19.16b, v27.16b\n\t"                      \
        "smmla v4.4s,  v20.16b, v28.16b\n\t"                      \
        "smmla v5.4s,  v21.16b, v29.16b\n\t"                      \
        "smmla v6.4s,  v22.16b, v30.16b\n\t"                      \
        "smmla v7.4s,  v23.16b, v31.16b\n\t"                      \
        "smmla v8.4s,  v16.16b, v24.16b\n\t"                      \
        "smmla v9.4s,  v17.16b, v25.16b\n\t"                      \
        "smmla v10.4s, v18.16b, v26.16b\n\t"                      \
        "smmla v11.4s, v19.16b, v27.16b\n\t"                      \
        "smmla v12.4s, v20.16b, v28.16b\n\t"                      \
        "smmla v13.4s, v21.16b, v29.16b\n\t"                      \
        "smmla v14.4s, v22.16b, v30.16b\n\t"                      \
        "smmla v15.4s, v23.16b, v31.16b"

    // ── Warmup + Best of 7 runs, fixed 15M iterations ──
    int64_t iters = 15000000LL;  // 15M * 1024 ops = 15.36 Gops, ~0.05s at 300 GOPS
    double best_gops = 0.0;
    for (int run = 0; run < 7; run++) {
        n = iters;
        clock_gettime(CLOCK_MONOTONIC_RAW, &s);
        __asm__ volatile (
            "movi v16.16b, #1\n\t"
            "movi v17.16b, #1\n\t"
            "movi v18.16b, #1\n\t"
            "movi v19.16b, #1\n\t"
            "movi v20.16b, #1\n\t"
            "movi v21.16b, #1\n\t"
            "movi v22.16b, #1\n\t"
            "movi v23.16b, #1\n\t"
            "movi v24.16b, #1\n\t"
            "movi v25.16b, #1\n\t"
            "movi v26.16b, #1\n\t"
            "movi v27.16b, #1\n\t"
            "movi v28.16b, #1\n\t"
            "movi v29.16b, #1\n\t"
            "movi v30.16b, #1\n\t"
            "movi v31.16b, #1\n\t"
            "movi v0.4s, #0\n\t"
            "movi v1.4s, #0\n\t"
            "movi v2.4s, #0\n\t"
            "movi v3.4s, #0\n\t"
            "movi v4.4s, #0\n\t"
            "movi v5.4s, #0\n\t"
            "movi v6.4s, #0\n\t"
            "movi v7.4s, #0\n\t"
            "movi v8.4s, #0\n\t"
            "movi v9.4s, #0\n\t"
            "movi v10.4s, #0\n\t"
            "movi v11.4s, #0\n\t"
            "movi v12.4s, #0\n\t"
            "movi v13.4s, #0\n\t"
            "movi v14.4s, #0\n\t"
            "movi v15.4s, #0\n\t"
            "1:\n\t"
            SMM_LOOP_BODY "\n\t"
            "subs %[n], %[n], #1\n\t"
            "b.ne 1b\n\t"
            : [n] "+r" (n)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
              "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
              "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
              "cc", "memory"
        );
        clock_gettime(CLOCK_MONOTONIC_RAW, &e);
        double elapsed = get_time(&s, &e);
        if (elapsed > 0.0) {
            double gops = iters * 1024.0 / elapsed * 1e-9;
            if (gops > best_gops) best_gops = gops;
        }
    }
    #undef SMM_LOOP_BODY
    return best_gops;
}

// ═══════════════════════════════════════════════════════════════════════
// BF16 kernels
// ═══════════════════════════════════════════════════════════════════════
typedef float    BF16_Accum;
typedef uint16_t BF16_Type;

static inline BF16_Type float_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return (BF16_Type)((u + ((u >> 16) & 1) + 0x7FFF) >> 16);
}

void bf16gemm_k_ld (const BF16_Type *A, const BF16_Type *B_reo,
                    BF16_Accum *C, int m, int k, int n,
                    BF16_Type *A_reorder, int lda, int ldb, int ldc);
void bf16gemm_k_ld1(const BF16_Type *A, const BF16_Type *B_reo,
                    BF16_Accum *C, int m, int k, int n,
                    BF16_Type *A_reorder, int lda, int ldb, int ldc);
void bf16gemm_k_ld2(const BF16_Type *A, const BF16_Type *B_reo,
                    BF16_Accum *C, int m, int k, int n,
                    BF16_Type *A_reorder, int lda, int ldb, int ldc);
void bf16gemm_k_ld4(const BF16_Type *A, const BF16_Type *B_reo,
                    BF16_Accum *C, int m, int k, int n,
                    BF16_Type *A_reorder, int lda, int ldb, int ldc);

static void bf16_pack_B(const BF16_Type* B, BF16_Type* B_reo, int K, int N) {
    int idx = 0;
    for (int cb = 0; cb < N/8; ++cb)
        for (int rb = 0; rb < K/4; ++rb) {
            int row_base = rb * 4, col_base = cb * 8;
            for (int cp = 0; cp < 4; ++cp) {
                int c0 = col_base + cp * 2, c1 = c0 + 1;
                for (int i = 0; i < 4; ++i)
                    B_reo[idx++] = B[(row_base + i) * N + c0];
                for (int i = 0; i < 4; ++i)
                    B_reo[idx++] = B[(row_base + i) * N + c1];
            }
        }
}

// ── Multi-threaded BF16 GEMM (from bf16gemm_mt.c) ──
void bf16_pack_B(const BF16_Type *B, BF16_Type *B_reo, int K, int N);
void bf16gemm_mt_dispatch(const BF16_Type *A, const BF16_Type *B_reo,
                           BF16_Accum *C, int M, int K_r, int N_r,
                           int num_threads);

static void bf16_dispatch(const BF16_Type *A, const BF16_Type *B_reo,
                           BF16_Accum *C, int m, int k, int n,
                           BF16_Type *A_reorder) {
    int lda = k, ldb = k, ldc = n;
    int processed = 0;
    int m_full = (m / 8) * 8;
    if (m_full > 0) {
        bf16gemm_k_ld(A, B_reo, C, m_full, k, n,
                      A_reorder, lda, ldb, ldc);
        processed = m_full;
    }
    int m_rem = m - processed;
    if (m_rem == 0) return;
    const BF16_Type *At = A + (uint64_t)processed * k;
    BF16_Accum      *Ct = C + (uint64_t)processed * n;
    BF16_Type *A_reo_t  = A_reorder + (uint64_t)processed * k;
    if (m_rem >= 4) {
        bf16gemm_k_ld4(At, B_reo, Ct, 4, k, n, A_reo_t, lda, ldb, ldc);
        processed += 4; m_rem -= 4;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * n;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 2) {
        bf16gemm_k_ld2(At, B_reo, Ct, 2, k, n, A_reo_t, lda, ldb, ldc);
        processed += 2; m_rem -= 2;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * n;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 1)
        bf16gemm_k_ld1(At, B_reo, Ct, 1, k, n, A_reo_t, lda, ldb, ldc);
}

// ═══════════════════════════════════════════════════════════════════════
// I8 kernels
// ═══════════════════════════════════════════════════════════════════════
typedef int32_t I8_Accum;
typedef int8_t  I8_Type;

void i8gemm_k_ld (const I8_Type *A, const I8_Type *B_reo,
                  I8_Accum *C, int m, int k, int n,
                  I8_Type *A_reorder, int lda, int ldb, int ldc);
void i8gemm_k_ld1(const I8_Type *A, const I8_Type *B_reo,
                  I8_Accum *C, int m, int k, int n,
                  I8_Type *A_reorder, int lda, int ldb, int ldc);
void i8gemm_k_ld2(const I8_Type *A, const I8_Type *B_reo,
                  I8_Accum *C, int m, int k, int n,
                  I8_Type *A_reorder, int lda, int ldb, int ldc);
void i8gemm_k_ld4(const I8_Type *A, const I8_Type *B_reo,
                  I8_Accum *C, int m, int k, int n,
                  I8_Type *A_reorder, int lda, int ldb, int ldc);

static void i8_pack_B(const I8_Type* B, I8_Type* B_reo, int K, int N) {
    int idx = 0;
    for (int cb = 0; cb < N/8; ++cb)
        for (int rb = 0; rb < K/8; ++rb)
            for (int j = 0; j < 8; ++j)
                for (int i = 0; i < 8; ++i)
                    B_reo[idx++] = B[(rb*8 + i) * N + (cb*8 + j)];
}

static void i8_dispatch(const I8_Type *A, const I8_Type *B_reo,
                         I8_Accum *C, int m, int k, int n,
                         I8_Type *A_reorder) {
    int lda = k, ldb = k, ldc = n;
    int processed = 0;
    int m_full = (m / 8) * 8;
    if (m_full > 0) {
        i8gemm_k_ld(A, B_reo, C, m_full, k, n,
                    A_reorder, lda, ldb, ldc);
        processed = m_full;
    }
    int m_rem = m - processed;
    if (m_rem == 0) return;
    const I8_Type *At = A + processed * k;
    I8_Accum      *Ct = C + processed * n;
    I8_Type *A_reo_t  = A_reorder + processed * k;
    if (m_rem >= 4) {
        i8gemm_k_ld4(At, B_reo, Ct, 4, k, n, A_reo_t, lda, ldb, ldc);
        processed += 4; m_rem -= 4;
        At = A + processed * k; Ct = C + processed * n;
        A_reo_t = A_reorder + processed * k;
    }
    if (m_rem >= 2) {
        i8gemm_k_ld2(At, B_reo, Ct, 2, k, n, A_reo_t, lda, ldb, ldc);
        processed += 2; m_rem -= 2;
        At = A + processed * k; Ct = C + processed * n;
        A_reo_t = A_reorder + processed * k;
    }
    if (m_rem >= 1)
        i8gemm_k_ld1(At, B_reo, Ct, 1, k, n, A_reo_t, lda, ldb, ldc);
}

// ── Multi-threaded I8 GEMM (from i8gemm_mt.c) ──
// Declared after static i8_pack_B/i8_dispatch to avoid redeclaration conflict
void i8gemm_mt_dispatch(const I8_Type *A, const I8_Type *B_reo,
                         I8_Accum *C, int M, int K_r, int N_r,
                         int num_threads);

// ═══════════════════════════════════════════════════════════════════════
// Auto-tune loops for benchmark
// ═══════════════════════════════════════════════════════════════════════
static int auto_loops(int m, int k, int n) {
    double flops = 2.0 * m * k * n;
    int loops = (int)(50e-3 * 200e9 / flops);
    if (loops < 1)  loops = 1;
    if (loops > 500) loops = 500;
    return loops;
}

// ═══════════════════════════════════════════════════════════════════════
// Benchmark one shape for both precisions
// ═══════════════════════════════════════════════════════════════════════
typedef struct {
    int m, k, n;
    double bf16_gflops, i8_gops;
    double bf16_eff, i8_eff;
} ShapeResult;

static ShapeResult bench_shape(int m, int k, int n, int loops) {
    ShapeResult r = {m, k, n, 0.0, 0.0, 0.0, 0.0};
    int bf_k_r = ((k + 7) / 8) * 8;   if (bf_k_r < 8)  bf_k_r = 8;
    int i8_k_r = ((k + 15) / 16) * 16; if (i8_k_r < 16) i8_k_r = 16;
    int n_r    = ((n + 7) / 8) * 8;   if (n_r < 8)   n_r = 8;

    srand(42);

    BF16_Type *bf_A     = (BF16_Type*)malloc((size_t)m * bf_k_r * sizeof(BF16_Type));
    BF16_Type *bf_B_orig = (BF16_Type*)malloc((size_t)bf_k_r * n_r * sizeof(BF16_Type));
    BF16_Type *bf_B_reo  = (BF16_Type*)malloc((size_t)bf_k_r * n_r * sizeof(BF16_Type));
    BF16_Accum *bf_C    = (BF16_Accum*)calloc((size_t)m * n_r, sizeof(BF16_Accum));
    BF16_Type *bf_A_reo = (BF16_Type*)malloc((size_t)(m + 8) * bf_k_r * sizeof(BF16_Type));

    I8_Type *i8_A     = (I8_Type*)malloc((size_t)m * i8_k_r);
    I8_Type *i8_B_orig = (I8_Type*)malloc((size_t)i8_k_r * n_r);
    I8_Type *i8_B_reo  = (I8_Type*)malloc((size_t)i8_k_r * n_r);
    I8_Accum *i8_C    = (I8_Accum*)calloc((size_t)m * n_r, sizeof(I8_Accum));
    I8_Type *i8_A_reo = (I8_Type*)malloc((size_t)(m + 8) * i8_k_r);

    if (!bf_A || !bf_B_orig || !bf_B_reo || !bf_C || !bf_A_reo ||
        !i8_A || !i8_B_orig || !i8_B_reo || !i8_C || !i8_A_reo)
        goto cleanup;

    for (int i = 0; i < m * bf_k_r; i++)
        bf_A[i] = float_to_bf16(((float)rand() / RAND_MAX) * 4.0f - 2.0f);
    for (int i = 0; i < bf_k_r * n_r; i++)
        bf_B_orig[i] = float_to_bf16(((float)rand() / RAND_MAX) * 4.0f - 2.0f);
    for (int i = 0; i < m * i8_k_r; i++)
        i8_A[i] = (I8_Type)(rand() % 256 - 128);
    for (int i = 0; i < i8_k_r * n_r; i++)
        i8_B_orig[i] = (I8_Type)(rand() % 256);

    bf16_pack_B(bf_B_orig, bf_B_reo, bf_k_r, n_r);
    i8_pack_B(i8_B_orig, i8_B_reo, i8_k_r, n_r);

    // Warmup
    for (int w = 0; w < 3; w++) {
        bf16_dispatch(bf_A, bf_B_reo, bf_C, m, bf_k_r, n_r, bf_A_reo);
        i8_dispatch(i8_A, i8_B_reo, i8_C, m, i8_k_r, n_r, i8_A_reo);
    }

    struct timespec s, e;

    clock_gettime(CLOCK_MONOTONIC_RAW, &s);
    for (int i = 0; i < loops; i++)
        bf16_dispatch(bf_A, bf_B_reo, bf_C, m, bf_k_r, n_r, bf_A_reo);
    clock_gettime(CLOCK_MONOTONIC_RAW, &e);
    double t = get_time(&s, &e);
    if (t > 0.0)
        r.bf16_gflops = (2.0 * m * bf_k_r * n_r) / (t / loops) * 1e-9;

    clock_gettime(CLOCK_MONOTONIC_RAW, &s);
    for (int i = 0; i < loops; i++)
        i8_dispatch(i8_A, i8_B_reo, i8_C, m, i8_k_r, n_r, i8_A_reo);
    clock_gettime(CLOCK_MONOTONIC_RAW, &e);
    t = get_time(&s, &e);
    if (t > 0.0)
        r.i8_gops = (2.0 * m * i8_k_r * n_r) / (t / loops) * 1e-9;

cleanup:
    free(bf_A); free(bf_B_orig); free(bf_B_reo); free(bf_C); free(bf_A_reo);
    free(i8_A); free(i8_B_orig); free(i8_B_reo); free(i8_C); free(i8_A_reo);
    return r;
}

// ═══════════════════════════════════════════════════════════════════════
// Sorting — by BF16 efficiency descending
// ═══════════════════════════════════════════════════════════════════════
static int cmp_eff_desc(const void *a, const void *b) {
    double ea = ((const ShapeResult*)a)->bf16_eff;
    double eb = ((const ShapeResult*)b)->bf16_eff;
    return (ea < eb) - (ea > eb);
}

// ═══════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════
// Multi-threaded BF16 GEMM benchmark (--mt mode)
// ═══════════════════════════════════════════════════════════════════════
static void bench_mt(int M, int K, int N, int num_threads) {
    int K_r = ((K + 7) / 8) * 8;   if (K_r < 8)  K_r = 8;
    int N_r = ((N + 7) / 8) * 8;   if (N_r < 8)  N_r = 8;

    const char *strategy = (M / 8 >= num_threads) ? "M-tiling" : "N-tiling";

    printf("# ═══ BF16 GEMM Multi-threaded ═══\n");
    printf("# Shape: M=%d K=%d N=%d  (K_r=%d N_r=%d)\n", M, K, N, K_r, N_r);
    printf("# Threads: %d  Strategy: %s\n", num_threads, strategy);
    printf("#\n");

    srand(42);
    BF16_Type *A = (BF16_Type*)malloc((size_t)M * K_r * sizeof(BF16_Type));
    BF16_Type *B = (BF16_Type*)malloc((size_t)K_r * N_r * sizeof(BF16_Type));
    BF16_Type *B_reo = (BF16_Type*)aligned_alloc(64,
        (size_t)K_r * N_r * sizeof(BF16_Type));
    BF16_Accum *C = (BF16_Accum*)calloc((size_t)M * N_r, sizeof(BF16_Accum));
    if (!A || !B || !B_reo || !C) {
        fprintf(stderr, "Alloc failed\n"); goto cleanup;
    }

    for (int i = 0; i < M * K_r; i++)
        A[i] = float_to_bf16(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    for (int i = 0; i < K_r * N_r; i++)
        B[i] = float_to_bf16(((float)rand() / RAND_MAX) * 2.0f - 1.0f);

    // Pack B once (outside timed region)
    bf16_pack_B(B, B_reo, K_r, N_r);

    // Warmup
    printf("# Warmup...\n");
    for (int w = 0; w < 3; w++)
        bf16gemm_mt_dispatch(A, B_reo, C, M, K_r, N_r, num_threads);

    // Benchmark multi-threaded
    printf("# Benchmarking...\n");
    double flops_per_call = 2.0 * (double)M * K_r * N_r;
    int loops = (int)(0.3 * 200e9 / flops_per_call);
    if (loops < 1) loops = 1;
    if (loops > 100) loops = 100;

    struct timespec s, e;
    double best_mt = 0.0;
    int runs = 5;
    for (int run = 0; run < runs; run++) {
        memset(C, 0, (size_t)M * N_r * sizeof(BF16_Accum));
        clock_gettime(CLOCK_MONOTONIC_RAW, &s);
        for (int i = 0; i < loops; i++)
            bf16gemm_mt_dispatch(A, B_reo, C, M, K_r, N_r, num_threads);
        clock_gettime(CLOCK_MONOTONIC_RAW, &e);
        double t = get_time(&s, &e);
        if (t > 0.0) {
            double gf = flops_per_call * loops / t * 1e-9;
            if (gf > best_mt) best_mt = gf;
        }
    }
    printf("#   Multi-threaded (%d threads, %s): %.1f GFLOPS  (loops=%d)\n",
           num_threads, strategy, best_mt, loops);

    // Compare with single-thread
    double best_st = 0.0;
    for (int run = 0; run < runs; run++) {
        memset(C, 0, (size_t)M * N_r * sizeof(BF16_Accum));
        clock_gettime(CLOCK_MONOTONIC_RAW, &s);
        for (int i = 0; i < loops; i++)
            bf16gemm_mt_dispatch(A, B_reo, C, M, K_r, N_r, 1);
        clock_gettime(CLOCK_MONOTONIC_RAW, &e);
        double t = get_time(&s, &e);
        if (t > 0.0) {
            double gf = flops_per_call * loops / t * 1e-9;
            if (gf > best_st) best_st = gf;
        }
    }
    printf("#   Single-thread (via mt_dispatch): %.1f GFLOPS\n", best_st);
    if (best_st > 0.0)
        printf("#   Speedup:                         %.2f x\n", best_mt / best_st);

    // Also report single-thread via original dispatch (for reference)
    BF16_Type *A_reo = (BF16_Type*)malloc((size_t)(M + 8) * K_r * sizeof(BF16_Type));
    BF16_Accum *C_st2 = (BF16_Accum*)calloc((size_t)M * N_r, sizeof(BF16_Accum));
    if (A_reo && C_st2) {
        for (int w = 0; w < 3; w++)
            bf16_dispatch(A, B_reo, C_st2, M, K_r, N_r, A_reo);
        double best_orig = 0.0;
        for (int run = 0; run < runs; run++) {
            memset(C_st2, 0, (size_t)M * N_r * sizeof(BF16_Accum));
            clock_gettime(CLOCK_MONOTONIC_RAW, &s);
            for (int i = 0; i < loops; i++)
                bf16_dispatch(A, B_reo, C_st2, M, K_r, N_r, A_reo);
            clock_gettime(CLOCK_MONOTONIC_RAW, &e);
            double t = get_time(&s, &e);
            if (t > 0.0) {
                double gf = flops_per_call * loops / t * 1e-9;
                if (gf > best_orig) best_orig = gf;
            }
        }
        printf("#   Single-thread (orig dispatch):   %.1f GFLOPS\n", best_orig);
    }
    free(A_reo); free(C_st2);

cleanup:
    free(A); free(B); free(B_reo); free(C);
}

// ═══════════════════════════════════════════════════════════════════════
// Multi-threaded I8 GEMM benchmark (--mt-i8 mode)
// ═══════════════════════════════════════════════════════════════════════
static void bench_mt_i8(int M, int K, int N, int num_threads) {
    int K_r = ((K + 15) / 16) * 16;  if (K_r < 16) K_r = 16;
    int N_r = ((N + 7)  / 8)  * 8;   if (N_r < 8)   N_r = 8;

    const char *strategy = (M / 8 >= num_threads) ? "M-tiling" : "N-tiling";

    printf("# ═══ I8 GEMM Multi-threaded ═══\n");
    printf("# Shape: M=%d K=%d N=%d  (K_r=%d N_r=%d)\n", M, K, N, K_r, N_r);
    printf("# Threads: %d  Strategy: %s\n", num_threads, strategy);
    printf("#\n");

    srand(42);
    I8_Type *A = (I8_Type*)malloc((size_t)M * K_r * sizeof(I8_Type));
    I8_Type *B = (I8_Type*)malloc((size_t)K_r * N_r * sizeof(I8_Type));
    I8_Type *B_reo = (I8_Type*)aligned_alloc(64,
        (size_t)K_r * N_r * sizeof(I8_Type));
    I8_Accum *C = (I8_Accum*)calloc((size_t)M * N_r, sizeof(I8_Accum));
    if (!A || !B || !B_reo || !C) {
        fprintf(stderr, "Alloc failed\n"); goto cleanup;
    }

    for (int i = 0; i < M * K_r; i++)
        A[i] = (I8_Type)(rand() % 256 - 128);
    for (int i = 0; i < K_r * N_r; i++)
        B[i] = (I8_Type)(rand() % 256);

    // Pack B once (outside timed region)
    i8_pack_B(B, B_reo, K_r, N_r);

    // Warmup
    printf("# Warmup...\n");
    for (int w = 0; w < 3; w++)
        i8gemm_mt_dispatch(A, B_reo, C, M, K_r, N_r, num_threads);

    // Benchmark multi-threaded
    printf("# Benchmarking...\n");
    double ops_per_call  = 2.0 * (double)M * K_r * N_r;
    double peak_gops = 331.6;  // approximate peak for timing estimate
    int loops = (int)(0.3 * peak_gops * 1e9 / ops_per_call);
    if (loops < 1) loops = 1;
    if (loops > 500) loops = 500;

    struct timespec s, e;
    double best_mt = 0.0;
    int runs = 5;
    for (int run = 0; run < runs; run++) {
        memset(C, 0, (size_t)M * N_r * sizeof(I8_Accum));
        clock_gettime(CLOCK_MONOTONIC_RAW, &s);
        for (int i = 0; i < loops; i++)
            i8gemm_mt_dispatch(A, B_reo, C, M, K_r, N_r, num_threads);
        clock_gettime(CLOCK_MONOTONIC_RAW, &e);
        double t = get_time(&s, &e);
        if (t > 0.0) {
            double go = ops_per_call * loops / t * 1e-9;
            if (go > best_mt) best_mt = go;
        }
    }
    printf("#   Multi-threaded (%d threads, %s): %.1f GOPS  (loops=%d)\n",
           num_threads, strategy, best_mt, loops);

    // Single-thread reference (via mt_dispatch)
    double best_st = 0.0;
    for (int run = 0; run < runs; run++) {
        memset(C, 0, (size_t)M * N_r * sizeof(I8_Accum));
        clock_gettime(CLOCK_MONOTONIC_RAW, &s);
        for (int i = 0; i < loops; i++)
            i8gemm_mt_dispatch(A, B_reo, C, M, K_r, N_r, 1);
        clock_gettime(CLOCK_MONOTONIC_RAW, &e);
        double t = get_time(&s, &e);
        if (t > 0.0) {
            double go = ops_per_call * loops / t * 1e-9;
            if (go > best_st) best_st = go;
        }
    }
    printf("#   Single-thread (via mt_dispatch): %.1f GOPS\n", best_st);
    if (best_st > 0.0)
        printf("#   Speedup:                         %.2f x\n", best_mt / best_st);

    // Single-thread via original dispatch (for cross-reference)
    I8_Type *A_reo = (I8_Type*)malloc((size_t)(M + 8) * K_r * sizeof(I8_Type));
    I8_Accum *C_st2 = (I8_Accum*)calloc((size_t)M * N_r, sizeof(I8_Accum));
    if (A_reo && C_st2) {
        for (int w = 0; w < 3; w++)
            i8_dispatch(A, B_reo, C_st2, M, K_r, N_r, A_reo);
        double best_orig = 0.0;
        for (int run = 0; run < runs; run++) {
            memset(C_st2, 0, (size_t)M * N_r * sizeof(I8_Accum));
            clock_gettime(CLOCK_MONOTONIC_RAW, &s);
            for (int i = 0; i < loops; i++)
                i8_dispatch(A, B_reo, C_st2, M, K_r, N_r, A_reo);
            clock_gettime(CLOCK_MONOTONIC_RAW, &e);
            double t = get_time(&s, &e);
            if (t > 0.0) {
                double go = ops_per_call * loops / t * 1e-9;
                if (go > best_orig) best_orig = go;
            }
        }
        printf("#   Single-thread (orig dispatch):   %.1f GOPS\n", best_orig);
    }
    free(A_reo); free(C_st2);

cleanup:
    free(A); free(B); free(B_reo); free(C);
}

// ═══════════════════════════════════════════════════════════════════════
// Multi-threaded BF16 + I8 GEMM benchmark (--mt-both mode)
// ═══════════════════════════════════════════════════════════════════════
static void bench_mt_both(int M, int K, int N, int num_threads) {
    printf("# ══════════════════════════════════════════════════════\n");
    printf("#  Multi-threaded BF16 + I8 GEMM\n");
    printf("#  Shape: M=%d K=%d N=%d  Threads: %d\n", M, K, N, num_threads);
    printf("# ══════════════════════════════════════════════════════\n\n");

    bench_mt(M, K, N, num_threads);
    printf("\n");
    bench_mt_i8(M, K, N, num_threads);
}

// ═══════════════════════════════════════════════════════════════════════
// Multi-threaded sweep (--mt-sweep / --mt-sweep-i8 / --mt-sweep-both)
//
// Sweeps thread counts {1,2,4,8,10,16,20,32,40,64}, skipping any
// count where nthreads >= ncores (no oversubscription).
// Sets OMP_PLACES=cores OMP_PROC_BIND=close for core affinity.
// ═══════════════════════════════════════════════════════════════════════
static int detect_ncores(void) {
    int nc = omp_get_num_procs();
    if (nc <= 0) nc = 64; // fallback
    return nc;
}

static void set_omp_affinity(void) {
    setenv("OMP_PLACES",  "cores", 1);
    setenv("OMP_PROC_BIND", "close", 1);
}

static void bench_mt_sweep_bf16(int M, int K, int N, int ncores) {
    int K_r = ((K + 7) / 8) * 8;   if (K_r < 8)  K_r = 8;
    int N_r = ((N + 7) / 8) * 8;   if (N_r < 8)  N_r = 8;

    srand(42);
    BF16_Type *A = (BF16_Type*)malloc((size_t)M * K_r * sizeof(BF16_Type));
    BF16_Type *B = (BF16_Type*)malloc((size_t)K_r * N_r * sizeof(BF16_Type));
    BF16_Type *B_reo = (BF16_Type*)aligned_alloc(64,
        (size_t)K_r * N_r * sizeof(BF16_Type));
    BF16_Accum *C = (BF16_Accum*)calloc((size_t)M * N_r, sizeof(BF16_Accum));
    if (!A || !B || !B_reo || !C) {
        fprintf(stderr, "Alloc failed\n"); goto cleanup;
    }

    for (int i = 0; i < M * K_r; i++)
        A[i] = float_to_bf16(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    for (int i = 0; i < K_r * N_r; i++)
        B[i] = float_to_bf16(((float)rand() / RAND_MAX) * 2.0f - 1.0f);

    bf16_pack_B(B, B_reo, K_r, N_r);

    double flops_per_call = 2.0 * (double)M * K_r * N_r;
    int default_threads[] = {1, 2, 4, 8, 10, 16, 20, 32, 40, 64};
    int ntests = sizeof(default_threads) / sizeof(default_threads[0]);

    int valid = 0;
    for (int t = 0; t < ntests; t++)
        if (default_threads[t] < ncores) valid++;

    printf("# ═══ BF16 Multi-threaded Sweep ═══\n");
    printf("# M=%d K=%d N=%d  (K_r=%d N_r=%d)  ncores=%d\n",
           M, K, N, K_r, N_r, ncores);
    printf("# Warmup...\n");

    // Warmup with max valid threads
    int max_t = default_threads[ntests - 1];
    while (max_t >= ncores && max_t > 1) max_t /= 2;
    for (int w = 0; w < 3; w++)
        bf16gemm_mt_dispatch(A, B_reo, C, M, K_r, N_r, max_t);

    printf("#\n");
    printf("# %6s  %10s  %8s  %9s\n",
           "nthr", "GFLOPS", "speedup", "efficiency");

    set_omp_affinity();

    double st_gflops = 0.0;
    for (int t = 0; t < ntests; t++) {
        int nth = default_threads[t];
        if (nth >= ncores) continue;

        double avg = 0.0;
        int runs = 5, ok = 0;
        for (int run = 0; run < runs; run++) {
            memset(C, 0, (size_t)M * N_r * sizeof(BF16_Accum));
            struct timespec s, e;
            clock_gettime(CLOCK_MONOTONIC_RAW, &s);
            bf16gemm_mt_dispatch(A, B_reo, C, M, K_r, N_r, nth);
            clock_gettime(CLOCK_MONOTONIC_RAW, &e);
            double elapsed = get_time(&s, &e);
            if (elapsed > 0.0) { avg += flops_per_call / elapsed * 1e-9; ok++; }
        }
        if (ok > 0) avg /= ok;
        if (nth == 1) st_gflops = avg;

        double speedup = (st_gflops > 0.0) ? avg / st_gflops : 1.0;
        printf("  %6d  %10.1f  %6.2fx  %9.1f%%\n",
               nth, avg, speedup, speedup / nth * 100.0);
    }

    printf("#\n");
    printf("# Efficiency = speedup / nthreads * 100%%\n");
    printf("# max_cores > nthreads for every row shown.\n");

cleanup:
    free(A); free(B); free(B_reo); free(C);
}

static void bench_mt_sweep_i8(int M, int K, int N, int ncores) {
    int K_r = ((K + 15) / 16) * 16;  if (K_r < 16) K_r = 16;
    int N_r = ((N + 7)  / 8)  * 8;   if (N_r < 8)   N_r = 8;

    srand(42);
    I8_Type *A = (I8_Type*)malloc((size_t)M * K_r * sizeof(I8_Type));
    I8_Type *B = (I8_Type*)malloc((size_t)K_r * N_r * sizeof(I8_Type));
    I8_Type *B_reo = (I8_Type*)aligned_alloc(64,
        (size_t)K_r * N_r * sizeof(I8_Type));
    I8_Accum *C = (I8_Accum*)calloc((size_t)M * N_r, sizeof(I8_Accum));
    if (!A || !B || !B_reo || !C) {
        fprintf(stderr, "Alloc failed\n"); goto cleanup;
    }

    for (int i = 0; i < M * K_r; i++)
        A[i] = (I8_Type)(rand() % 256 - 128);
    for (int i = 0; i < K_r * N_r; i++)
        B[i] = (I8_Type)(rand() % 256);

    i8_pack_B(B, B_reo, K_r, N_r);

    double ops_per_call = 2.0 * (double)M * K_r * N_r;
    int default_threads[] = {1, 2, 4, 8, 10, 16, 20, 32, 40, 64};
    int ntests = sizeof(default_threads) / sizeof(default_threads[0]);

    int valid = 0;
    for (int t = 0; t < ntests; t++)
        if (default_threads[t] < ncores) valid++;

    printf("# ═══ I8 Multi-threaded Sweep ═══\n");
    printf("# M=%d K=%d N=%d  (K_r=%d N_r=%d)  ncores=%d\n",
           M, K, N, K_r, N_r, ncores);
    printf("# Warmup...\n");

    int max_t = default_threads[ntests - 1];
    while (max_t >= ncores && max_t > 1) max_t /= 2;
    for (int w = 0; w < 3; w++)
        i8gemm_mt_dispatch(A, B_reo, C, M, K_r, N_r, max_t);

    printf("#\n");
    printf("# %6s  %10s  %8s  %9s\n",
           "nthr", "GOPS", "speedup", "efficiency");

    set_omp_affinity();

    double st_gops = 0.0;
    for (int t = 0; t < ntests; t++) {
        int nth = default_threads[t];
        if (nth >= ncores) continue;

        double avg = 0.0;
        int runs = 5, ok = 0;
        for (int run = 0; run < runs; run++) {
            memset(C, 0, (size_t)M * N_r * sizeof(I8_Accum));
            struct timespec s, e;
            clock_gettime(CLOCK_MONOTONIC_RAW, &s);
            i8gemm_mt_dispatch(A, B_reo, C, M, K_r, N_r, nth);
            clock_gettime(CLOCK_MONOTONIC_RAW, &e);
            double elapsed = get_time(&s, &e);
            if (elapsed > 0.0) { avg += ops_per_call / elapsed * 1e-9; ok++; }
        }
        if (ok > 0) avg /= ok;
        if (nth == 1) st_gops = avg;

        double speedup = (st_gops > 0.0) ? avg / st_gops : 1.0;
        printf("  %6d  %10.1f  %6.2fx  %9.1f%%\n",
               nth, avg, speedup, speedup / nth * 100.0);
    }

    printf("#\n");
    printf("# Efficiency = speedup / nthreads * 100%%\n");
    printf("# max_cores > nthreads for every row shown.\n");

cleanup:
    free(A); free(B); free(B_reo); free(C);
}

static void bench_mt_sweep_both(int M, int K, int N, int ncores) {
    bench_mt_sweep_bf16(M, K, N, ncores);
    printf("\n");
    bench_mt_sweep_i8(M, K, N, ncores);
}

// ═══════════════════════════════════════════════════════════════════════
// CSV-based sweep helpers: benchmark one shape, return results array.
// caller frees with free().
// ═══════════════════════════════════════════════════════════════════════
typedef struct {
    int nthr;
    double gflops;   // or gops for i8
} SweepPoint;

static SweepPoint* sweep_one_bf16(int M, int K, int N, int ncores, int *nout) {
    int K_r = ((K + 7) / 8) * 8;   if (K_r < 8)  K_r = 8;
    int N_r = ((N + 7) / 8) * 8;   if (N_r < 8)  N_r = 8;
    int default_threads[] = {1, 2, 4, 8, 10, 16, 20, 32, 40, 64};
    int ntests = sizeof(default_threads) / sizeof(default_threads[0]);

    // count valid
    int nvalid = 0;
    for (int t = 0; t < ntests; t++)
        if (default_threads[t] < ncores) nvalid++;
    SweepPoint *pts = (SweepPoint*)calloc(nvalid, sizeof(SweepPoint));
    if (!pts) { *nout = 0; return NULL; }

    BF16_Type *A     = (BF16_Type*)malloc((size_t)M * K_r * sizeof(BF16_Type));
    BF16_Type *B     = (BF16_Type*)malloc((size_t)K_r * N_r * sizeof(BF16_Type));
    BF16_Type *B_reo = (BF16_Type*)aligned_alloc(64, (size_t)K_r * N_r * sizeof(BF16_Type));
    BF16_Accum *C    = (BF16_Accum*)calloc((size_t)M * N_r, sizeof(BF16_Accum));
    if (!A || !B || !B_reo || !C) goto done;

    srand(42);
    for (int i = 0; i < M * K_r; i++)
        A[i] = float_to_bf16(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    for (int i = 0; i < K_r * N_r; i++)
        B[i] = float_to_bf16(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    bf16_pack_B(B, B_reo, K_r, N_r);
    double flops = 2.0 * (double)M * K_r * N_r;

    // warmup with max valid threads
    int max_t = default_threads[ntests - 1];
    while (max_t >= ncores && max_t > 1) max_t /= 2;
    for (int w = 0; w < 2; w++)
        bf16gemm_mt_dispatch(A, B_reo, C, M, K_r, N_r, max_t);

    int idx = 0;
    for (int t = 0; t < ntests; t++) {
        int nth = default_threads[t];
        if (nth >= ncores) continue;

        double avg = 0.0; int ok = 0;
        for (int run = 0; run < 3; run++) {
            memset(C, 0, (size_t)M * N_r * sizeof(BF16_Accum));
            struct timespec s, e;
            clock_gettime(CLOCK_MONOTONIC_RAW, &s);
            bf16gemm_mt_dispatch(A, B_reo, C, M, K_r, N_r, nth);
            clock_gettime(CLOCK_MONOTONIC_RAW, &e);
            double t = get_time(&s, &e);
            if (t > 0.0) { avg += flops / t * 1e-9; ok++; }
        }
        if (ok > 0) avg /= ok;
        pts[idx].nthr   = nth;
        pts[idx].gflops = avg;
        idx++;
    }
    *nout = idx;

done:
    free(A); free(B); free(B_reo); free(C);
    return pts;
}

static SweepPoint* sweep_one_i8(int M, int K, int N, int ncores, int *nout) {
    int K_r = ((K + 15) / 16) * 16;  if (K_r < 16) K_r = 16;
    int N_r = ((N + 7)  / 8)  * 8;   if (N_r < 8)   N_r = 8;
    int default_threads[] = {1, 2, 4, 8, 10, 16, 20, 32, 40, 64};
    int ntests = sizeof(default_threads) / sizeof(default_threads[0]);

    int nvalid = 0;
    for (int t = 0; t < ntests; t++)
        if (default_threads[t] < ncores) nvalid++;
    SweepPoint *pts = (SweepPoint*)calloc(nvalid, sizeof(SweepPoint));
    if (!pts) { *nout = 0; return NULL; }

    I8_Type *A     = (I8_Type*)malloc((size_t)M * K_r * sizeof(I8_Type));
    I8_Type *B     = (I8_Type*)malloc((size_t)K_r * N_r * sizeof(I8_Type));
    I8_Type *B_reo = (I8_Type*)aligned_alloc(64, (size_t)K_r * N_r * sizeof(I8_Type));
    I8_Accum *C    = (I8_Accum*)calloc((size_t)M * N_r, sizeof(I8_Accum));
    if (!A || !B || !B_reo || !C) goto done;

    srand(42);
    for (int i = 0; i < M * K_r; i++)
        A[i] = (I8_Type)(rand() % 256 - 128);
    for (int i = 0; i < K_r * N_r; i++)
        B[i] = (I8_Type)(rand() % 256);
    i8_pack_B(B, B_reo, K_r, N_r);
    double ops = 2.0 * (double)M * K_r * N_r;

    int max_t = default_threads[ntests - 1];
    while (max_t >= ncores && max_t > 1) max_t /= 2;
    for (int w = 0; w < 2; w++)
        i8gemm_mt_dispatch(A, B_reo, C, M, K_r, N_r, max_t);

    int idx = 0;
    for (int t = 0; t < ntests; t++) {
        int nth = default_threads[t];
        if (nth >= ncores) continue;

        double avg = 0.0; int ok = 0;
        for (int run = 0; run < 3; run++) {
            memset(C, 0, (size_t)M * N_r * sizeof(I8_Accum));
            struct timespec s, e;
            clock_gettime(CLOCK_MONOTONIC_RAW, &s);
            i8gemm_mt_dispatch(A, B_reo, C, M, K_r, N_r, nth);
            clock_gettime(CLOCK_MONOTONIC_RAW, &e);
            double t = get_time(&s, &e);
            if (t > 0.0) { avg += ops / t * 1e-9; ok++; }
        }
        if (ok > 0) avg /= ok;
        pts[idx].nthr   = nth;
        pts[idx].gflops = avg;   // reuse gflops field for gops
        idx++;
    }
    *nout = idx;

done:
    free(A); free(B); free(B_reo); free(C);
    return pts;
}

// ═══════════════════════════════════════════════════════════════════════
// CSV-based multi-threaded sweeps — read shapes from CSV, thread-sweep each.
// Output: CSV lines with columns M,K,N,nthr,GFLOPS,speedup,efficiency
// ═══════════════════════════════════════════════════════════════════════

static void bench_mt_sweep_csv_bf16(const char *csv_path, int ncores) {
    FILE *fp = fopen(csv_path, "r");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", csv_path); return; }

    // Pre-count shapes for progress
    char line[256];
    int total = 0;
    while (fgets(line, sizeof(line), fp)) {
        int m, k, n;
        if (sscanf(line, "%d,%d,%d", &m, &k, &n) == 3 && m > 0 && k > 0 && n > 0)
            total++;
    }
    rewind(fp);

    int default_threads[] = {1, 2, 4, 8, 10, 16, 20, 32, 40, 64};
    int ntests = sizeof(default_threads) / sizeof(default_threads[0]);

    set_omp_affinity();

    printf("# bf16 mt-sweep-csv  ncores=%d  shapes=%d\n", ncores, total);
    printf("# %6s,%6s,%6s", "M", "K", "N");
    for (int t = 0; t < ntests; t++)
        if (default_threads[t] < ncores)
            printf(",%dgflops_t%d", default_threads[t], default_threads[t]);
    printf("\n");

    int done = 0;
    while (fgets(line, sizeof(line), fp)) {
        int m, k, n;
        if (sscanf(line, "%d,%d,%d", &m, &k, &n) != 3 || m <= 0 || k <= 0 || n <= 0)
            continue;
        done++;
        fprintf(stderr, "\r  sweep-csv bf16 %d/%d (%d,%d,%d)    ", done, total, m, k, n);

        int nout = 0;
        SweepPoint *pts = sweep_one_bf16(m, k, n, ncores, &nout);
        if (!pts) continue;

        printf("  %6d,%6d,%6d", m, k, n);
        for (int i = 0; i < nout; i++)
            printf(",%.1f", pts[i].gflops);
        printf("\n");
        free(pts);
    }
    fprintf(stderr, "\n");
    fclose(fp);
}

static void bench_mt_sweep_csv_i8(const char *csv_path, int ncores) {
    FILE *fp = fopen(csv_path, "r");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", csv_path); return; }

    char line[256];
    int total = 0;
    while (fgets(line, sizeof(line), fp)) {
        int m, k, n;
        if (sscanf(line, "%d,%d,%d", &m, &k, &n) == 3 && m > 0 && k > 0 && n > 0)
            total++;
    }
    rewind(fp);

    int default_threads[] = {1, 2, 4, 8, 10, 16, 20, 32, 40, 64};
    int ntests = sizeof(default_threads) / sizeof(default_threads[0]);

    set_omp_affinity();

    printf("# i8 mt-sweep-csv  ncores=%d  shapes=%d\n", ncores, total);
    printf("# %6s,%6s,%6s", "M", "K", "N");
    for (int t = 0; t < ntests; t++)
        if (default_threads[t] < ncores)
            printf(",%dgops_t%d", default_threads[t], default_threads[t]);
    printf("\n");

    int done = 0;
    while (fgets(line, sizeof(line), fp)) {
        int m, k, n;
        if (sscanf(line, "%d,%d,%d", &m, &k, &n) != 3 || m <= 0 || k <= 0 || n <= 0)
            continue;
        done++;
        fprintf(stderr, "\r  sweep-csv i8 %d/%d (%d,%d,%d)    ", done, total, m, k, n);

        int nout = 0;
        SweepPoint *pts = sweep_one_i8(m, k, n, ncores, &nout);
        if (!pts) continue;

        printf("  %6d,%6d,%6d", m, k, n);
        for (int i = 0; i < nout; i++)
            printf(",%.1f", pts[i].gflops);
        printf("\n");
        free(pts);
    }
    fprintf(stderr, "\n");
    fclose(fp);
}

static void bench_mt_sweep_csv_both(const char *csv_path, int ncores) {
    bench_mt_sweep_csv_bf16(csv_path, ncores);
    printf("\n");
    bench_mt_sweep_csv_i8(csv_path, ncores);
}

// ═══════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════
int main(int argc, char *argv[]) {
    // ── CSV sweep modes (read shapes from CSV, thread-sweep each) ──
    if (argc >= 2 && strcmp(argv[1], "--mt-sweep-csv") == 0) {
        const char *csv = (argc >= 3) ? argv[2] : "shape.csv";
        int ncores = detect_ncores();
        bench_mt_sweep_csv_bf16(csv, ncores);
        return 0;
    }
    if (argc >= 2 && strcmp(argv[1], "--mt-sweep-csv-i8") == 0) {
        const char *csv = (argc >= 3) ? argv[2] : "shape.csv";
        int ncores = detect_ncores();
        bench_mt_sweep_csv_i8(csv, ncores);
        return 0;
    }
    if (argc >= 2 && strcmp(argv[1], "--mt-sweep-csv-both") == 0) {
        const char *csv = (argc >= 3) ? argv[2] : "shape.csv";
        int ncores = detect_ncores();
        bench_mt_sweep_csv_both(csv, ncores);
        return 0;
    }

    // ── Multi-threaded sweep modes ──
    if (argc >= 2 && strcmp(argv[1], "--mt-sweep") == 0) {
        if (argc < 5) {
            fprintf(stderr, "Usage: %s --mt-sweep M K N\n", argv[0]);
            return 1;
        }
        int M = atoi(argv[2]), K = atoi(argv[3]), N = atoi(argv[4]);
        if (M <= 0 || K <= 0 || N <= 0) {
            fprintf(stderr, "Invalid arguments\n");
            return 1;
        }
        int ncores = detect_ncores();
        bench_mt_sweep_bf16(M, K, N, ncores);
        return 0;
    }

    if (argc >= 2 && strcmp(argv[1], "--mt-sweep-i8") == 0) {
        if (argc < 5) {
            fprintf(stderr, "Usage: %s --mt-sweep-i8 M K N\n", argv[0]);
            return 1;
        }
        int M = atoi(argv[2]), K = atoi(argv[3]), N = atoi(argv[4]);
        if (M <= 0 || K <= 0 || N <= 0) {
            fprintf(stderr, "Invalid arguments\n");
            return 1;
        }
        int ncores = detect_ncores();
        bench_mt_sweep_i8(M, K, N, ncores);
        return 0;
    }

    if (argc >= 2 && strcmp(argv[1], "--mt-sweep-both") == 0) {
        if (argc < 5) {
            fprintf(stderr, "Usage: %s --mt-sweep-both M K N\n", argv[0]);
            return 1;
        }
        int M = atoi(argv[2]), K = atoi(argv[3]), N = atoi(argv[4]);
        if (M <= 0 || K <= 0 || N <= 0) {
            fprintf(stderr, "Invalid arguments\n");
            return 1;
        }
        int ncores = detect_ncores();
        bench_mt_sweep_both(M, K, N, ncores);
        return 0;
    }

    // ── Multi-threaded modes ──
    if (argc >= 2 && strcmp(argv[1], "--mt") == 0) {
        if (argc < 5) {
            fprintf(stderr, "Usage: %s --mt M K N [nthreads]\n", argv[0]);
            return 1;
        }
        int M = atoi(argv[2]), K = atoi(argv[3]), N = atoi(argv[4]);
        int nthreads = (argc >= 6) ? atoi(argv[5]) : omp_get_max_threads();
        if (M <= 0 || K <= 0 || N <= 0 || nthreads <= 0) {
            fprintf(stderr, "Invalid arguments\n");
            return 1;
        }
        bench_mt(M, K, N, nthreads);
        return 0;
    }

    if (argc >= 2 && strcmp(argv[1], "--mt-i8") == 0) {
        if (argc < 5) {
            fprintf(stderr, "Usage: %s --mt-i8 M K N [nthreads]\n", argv[0]);
            return 1;
        }
        int M = atoi(argv[2]), K = atoi(argv[3]), N = atoi(argv[4]);
        int nthreads = (argc >= 6) ? atoi(argv[5]) : omp_get_max_threads();
        if (M <= 0 || K <= 0 || N <= 0 || nthreads <= 0) {
            fprintf(stderr, "Invalid arguments\n");
            return 1;
        }
        bench_mt_i8(M, K, N, nthreads);
        return 0;
    }

    if (argc >= 2 && strcmp(argv[1], "--mt-both") == 0) {
        if (argc < 5) {
            fprintf(stderr, "Usage: %s --mt-both M K N [nthreads]\n", argv[0]);
            return 1;
        }
        int M = atoi(argv[2]), K = atoi(argv[3]), N = atoi(argv[4]);
        int nthreads = (argc >= 6) ? atoi(argv[5]) : omp_get_max_threads();
        if (M <= 0 || K <= 0 || N <= 0 || nthreads <= 0) {
            fprintf(stderr, "Invalid arguments\n");
            return 1;
        }
        bench_mt_both(M, K, N, nthreads);
        return 0;
    }

    const char *csv_path = (argc >= 2) ? argv[1] : "shape.csv";

    // ── 1. Measure peak throughput ──
    printf("# ═══ Peak Instruction Throughput ═══\n");
    fflush(stdout);

    double peak_bf16 = measure_peak_bfmmla();
    printf("# Peak BF16 (bfmmla):  %.1f GFLOPS\n", peak_bf16);

    double peak_i8 = measure_peak_smmla();
    printf("# Peak I8   (smmla):   %.1f GOPS\n", peak_i8);
    printf("#\n");
    fflush(stdout);

    if (peak_bf16 <= 0.0) peak_bf16 = 1.0;
    if (peak_i8 <= 0.0)   peak_i8   = 1.0;

    // ── 2. Read shapes ──
    FILE *fp = fopen(csv_path, "r");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", csv_path); return 1; }

    char line[256];
    int num_shapes = 0;
    while (fgets(line, sizeof(line), fp)) {
        int m = 0, k = 0, n = 0;
        if (sscanf(line, "%d,%d,%d", &m, &k, &n) == 3 && m > 0 && k > 0 && n > 0)
            num_shapes++;
    }

    ShapeResult *results = (ShapeResult*)calloc(num_shapes, sizeof(ShapeResult));
    if (!results) { fclose(fp); return 1; }

    rewind(fp);
    int idx = 0, done = 0;
    while (fgets(line, sizeof(line), fp) && idx < num_shapes) {
        int m = 0, k = 0, n = 0;
        if (sscanf(line, "%d,%d,%d", &m, &k, &n) != 3 || m <= 0 || k <= 0 || n <= 0)
            continue;
        // Show progress
        done++;
        fprintf(stderr, "\r  benchmarking %d/%d: (%d,%d,%d)    ", done, num_shapes, m, k, n);
        fflush(stderr);

        int loops = auto_loops(m, k, n);
        results[idx] = bench_shape(m, k, n, loops);
        idx++;
    }
    fclose(fp);
    num_shapes = idx;
    fprintf(stderr, "\n");

    // ── 3. Compute efficiency and sort ──
    for (int i = 0; i < num_shapes; i++) {
        results[i].bf16_eff = 100.0 * results[i].bf16_gflops / peak_bf16;
        results[i].i8_eff   = 100.0 * results[i].i8_gops   / peak_i8;
    }
    qsort(results, num_shapes, sizeof(ShapeResult), cmp_eff_desc);

    // ── 4. Output ──
    printf("# ═══ Shape Benchmarks (sorted by BF16 efficiency, descending) ═══\n");
    printf("# %38s  %10s  %10s  %9s  %9s\n",
           "Shape(M,K,N)", "bf16_eff%", "bf16_GFLOPS", "i8_eff%", "i8_GOPS");
    printf("# %38s  %10s  %10s  %9s  %9s\n",
           "--------------------------------------", "----------", "----------", "---------", "---------");

    for (int i = 0; i < num_shapes; i++) {
        printf("(%6d,%6d,%6d)  %9.1f%%  %10.1f  %8.1f%%  %9.1f\n",
               results[i].m, results[i].k, results[i].n,
               results[i].bf16_eff, results[i].bf16_gflops,
               results[i].i8_eff, results[i].i8_gops);
    }

    // ── 5. Summary statistics ──
    double bf_sum = 0, bf_min = 1000, bf_max = 0;
    double i8_sum = 0, i8_min = 1000, i8_max = 0;
    for (int i = 0; i < num_shapes; i++) {
        bf_sum += results[i].bf16_eff;
        if (results[i].bf16_eff < bf_min) bf_min = results[i].bf16_eff;
        if (results[i].bf16_eff > bf_max) bf_max = results[i].bf16_eff;
        i8_sum += results[i].i8_eff;
        if (results[i].i8_eff < i8_min) i8_min = results[i].i8_eff;
        if (results[i].i8_eff > i8_max) i8_max = results[i].i8_eff;
    }
    printf("#\n");
    printf("# Summary:\n");
    printf("#   BF16 efficiency: avg=%.1f%%, min=%.1f%%, max=%.1f%%\n",
           num_shapes > 0 ? bf_sum / num_shapes : 0, bf_min, bf_max);
    printf("#   I8   efficiency: avg=%.1f%%, min=%.1f%%, max=%.1f%%\n",
           num_shapes > 0 ? i8_sum / num_shapes : 0, i8_min, i8_max);

    free(results);
    return 0;
}
