// bench_perf.c
// Unified benchmark: peak instruction throughput (bfmmla/smmla) +
// shape.csv performance test + efficiency ranking.
//
// Build:
//   cc -o bench_perf bench_perf.c \
//      i8gemm_k.S bf16gemm_k.S \
//      -march=armv8.6-a+bf16+i8mm -O2 -Wall -lm
//
// Usage:
//   ./bench_perf [shape.csv]  (default: shape.csv)
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
int main(int argc, char *argv[]) {
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
