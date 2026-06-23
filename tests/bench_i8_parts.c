// bench_i8_parts.c -- microbenchmark one M8 I8 kernel attribution variant.

#define _GNU_SOURCE
#include <arm_sve.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef I8_BENCH_NTILE
#define I8_BENCH_NTILE 0
#endif

#ifndef I8_BENCH_SVE_PACK
#define I8_BENCH_SVE_PACK 0
#endif

typedef struct {
    int m;
    int k;
    int n;
    int lda;
    int ldb;
    int ldc;
} gemm_params_t;

extern void i8gemm_k_ld(const int8_t *A, const int8_t *B_reo,
                        int32_t *C, int8_t *A_reorder,
                        const gemm_params_t *params);
extern void i8gemm_k_reo_ld(const int8_t *A, const int8_t *B_reo,
                            int32_t *C, int8_t *A_reorder,
                            const gemm_params_t *params);

static int g_prepacked_a = 0;

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static size_t round64(size_t n) {
    return (n + 63u) & ~(size_t)63u;
}

static void *xalloc(size_t n) {
    if (n == 0)
        n = 64;
    void *p = aligned_alloc(64, round64(n));
    if (!p) {
        perror("aligned_alloc");
        exit(1);
    }
    memset(p, 1, round64(n));
    return p;
}

static const char *cache_class(double kib) {
    if (kib <= 96.0)
        return "L1";
    if (kib <= 640.0)
        return "H2";
    if (kib <= 1280.0)
        return "L2";
    return "GT_L2";
}

static int i8_bench_n_tile(void) {
#if I8_BENCH_NTILE > 0
    return I8_BENCH_NTILE;
#else
    return (int)(svcntb() / 16) * 8;
#endif
}

static void pack_a8_i8(const int8_t *A, int8_t *P, int M, int K, int lda) {
    size_t idx = 0;
    for (int mb = 0; mb < M; mb += 8) {
        for (int kb = 0; kb < K; kb += 16) {
            for (int half = 0; half < 16; half += 8) {
                for (int rp = 0; rp < 4; rp++) {
                    int r0 = mb + rp * 2;
                    int r1 = r0 + 1;
                    memcpy(P + idx, A + (size_t)r0 * (size_t)lda + kb + half, 8);
                    idx += 8;
                    memcpy(P + idx, A + (size_t)r1 * (size_t)lda + kb + half, 8);
                    idx += 8;
                }
            }
        }
    }
}

static void pack_b_neon_i8(const int8_t *B, int8_t *B_reo, int K, int N,
                           int ldb) {
    size_t idx = 0;
    for (int cb = 0; cb < N / 8; cb++) {
        for (int rb = 0; rb < K / 8; rb++) {
            for (int j = 0; j < 8; j++) {
                for (int i = 0; i < 8; i++)
                    B_reo[idx++] = B[(size_t)(rb * 8 + i) * (size_t)ldb +
                                      (cb * 8 + j)];
            }
        }
    }
}

static void pack_b_sve_i8(const int8_t *B, int8_t *B_reo, int K, int N,
                          int ldb) {
    const int segs = i8_bench_n_tile() / 8;
    const int n_tile = segs * 8;
    size_t idx = 0;
    for (int nb = 0; nb < N; nb += n_tile) {
        for (int rb = 0; rb < K / 8; rb++) {
            int row_base = rb * 8;
            for (int cp = 0; cp < 4; cp++) {
                for (int sg = 0; sg < segs; sg++) {
                    int col_base = nb + sg * 8 + cp * 2;
                    for (int j = 0; j < 2; j++) {
                        for (int i = 0; i < 8; i++)
                            B_reo[idx++] =
                                B[(size_t)(row_base + i) * (size_t)ldb +
                                  col_base + j];
                    }
                }
            }
        }
    }
}

static int use_n_split_for_shape(int M, int K, int N, int threads) {
    if (threads <= 1)
        return 0;

    const char *split = getenv("M8_PARTS_SPLIT");
    int n_tile = i8_bench_n_tile();
    int n_tiles = N / n_tile;
    int m_blocks = M / 8;

    if (split) {
        if (strcmp(split, "m") == 0)
            return 0;
        if (strcmp(split, "n") == 0)
            return n_tiles >= threads;
        if (strcmp(split, "auto") != 0)
            return 0;
    }

    if (n_tiles < threads)
        return 0;
    if (m_blocks < threads)
        return 1;

    size_t b_panel_bytes = (size_t)K * (size_t)N;
    return b_panel_bytes >= 512u * 1024u && N >= M * 2;
}

static void run_kernel(const int8_t *A, const int8_t *B, int32_t *C,
                       int8_t *A_reorder, const gemm_params_t *params) {
    if (g_prepacked_a)
        i8gemm_k_reo_ld(A, B, C, A_reorder, params);
    else
        i8gemm_k_ld(A, B, C, A_reorder, params);
}

static void run_kernel_mt(const int8_t *A, const int8_t *B, int32_t *C,
                          int8_t *A_reorder, const gemm_params_t *params,
                          int threads) {
    if (threads <= 1) {
        run_kernel(A, B, C, A_reorder, params);
        return;
    }

    if (use_n_split_for_shape(params->m, params->k, params->n, threads)) {
        int n_tile = i8_bench_n_tile();
        int n_tiles = params->n / n_tile;
#pragma omp parallel for num_threads(threads) schedule(static)
        for (int t = 0; t < n_tiles; t++) {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            size_t reorder_stride = (size_t)params->m * (size_t)params->k;
            gemm_params_t local = *params;
            local.n = n_tile;
            run_kernel(A,
                       B + (size_t)t * (size_t)params->k * (size_t)n_tile,
                       C + (size_t)t * (size_t)n_tile,
                       A_reorder + (size_t)tid * reorder_stride,
                       &local);
        }
        return;
    }

    int m_blocks = params->m / 8;
#pragma omp parallel num_threads(threads)
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
#else
        int tid = 0;
        int nth = 1;
#endif
        int block_start = (m_blocks * tid) / nth;
        int block_end = (m_blocks * (tid + 1)) / nth;
        int m_start = block_start * 8;
        int m_count = (block_end - block_start) * 8;
        if (m_count > 0) {
            gemm_params_t local = *params;
            local.m = m_count;
            run_kernel(A + (size_t)m_start * (size_t)params->lda,
                       B,
                       C + (size_t)m_start * (size_t)params->ldc,
                       A_reorder + (size_t)m_start * (size_t)params->k,
                       &local);
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 8 || argc > 12) {
        fprintf(stderr,
                "usage: %s variant M K N reps warmup runs [i32] [threads] [stride_factor] [batch_count]\n",
                argv[0]);
        return 2;
    }

    const char *variant = argv[1];
    g_prepacked_a = strstr(variant, "prepacked") != NULL;
    int M = atoi(argv[2]);
    int K = atoi(argv[3]);
    int N = atoi(argv[4]);
    int reps = atoi(argv[5]);
    int warmup = atoi(argv[6]);
    int runs = atoi(argv[7]);
    const char *mode = argc >= 9 ? argv[8] : "i32";
    int threads = argc >= 10 ? atoi(argv[9]) : 1;
    int stride_factor = argc >= 11 ? atoi(argv[10]) : 1;
    int batch_count = argc >= 12 ? atoi(argv[11]) : 1;

    if (strcmp(mode, "i32") != 0) {
        fprintf(stderr, "bad mode: %s\n", mode);
        return 2;
    }
    if (threads < 1 || stride_factor < 1 || batch_count < 1)
        return 2;
    if (M % 8 != 0) {
        fprintf(stderr, "M must be a multiple of 8 for this M8 bench: %d\n", M);
        return 2;
    }
    if (K % 16 != 0 || N % i8_bench_n_tile() != 0) {
        fprintf(stderr, "K/N must be aligned for this I8 M8 bench: K=%d N=%d tile=%d\n",
                K, N, i8_bench_n_tile());
        return 2;
    }

    const double peak_gops = 660.0;
    int lda = K * stride_factor;
    int ldb = N * stride_factor;
    int ldc = N * stride_factor;
    size_t a_elems = (size_t)M * (size_t)lda;
    size_t b_elems = (size_t)K * (size_t)ldb;
    size_t c_elems = (size_t)M * (size_t)ldc;
    size_t a_bytes = a_elems;
    size_t b_bytes = b_elems;
    size_t b_reo_bytes = (size_t)K * (size_t)N;
    size_t c_bytes = c_elems * sizeof(int32_t);
    size_t reorder_copies = (size_t)threads;
    if (reorder_copies < 1)
        reorder_copies = 1;
    size_t reorder_elems = (size_t)M * (size_t)K * reorder_copies;
    size_t one_batch_bytes = a_bytes + b_reo_bytes + c_bytes;
    double kib = (double)one_batch_bytes * (double)batch_count / 1024.0;

    int8_t *A = (int8_t *)xalloc(a_bytes * (size_t)batch_count);
    int8_t *B = (int8_t *)xalloc(b_bytes * (size_t)batch_count);
    int8_t *B_reo = (int8_t *)xalloc(b_reo_bytes * (size_t)batch_count);
    int8_t *A_reorder = (int8_t *)xalloc(reorder_elems * (size_t)batch_count);
    int32_t *C = (int32_t *)xalloc(c_bytes * (size_t)batch_count);

    for (int b = 0; b < batch_count; b++) {
        int8_t *batch_a = A + (size_t)b * a_elems;
        int8_t *batch_b = B + (size_t)b * b_elems;
        int8_t *batch_b_reo = B_reo + (size_t)b * b_reo_bytes;
        int8_t *batch_reorder = A_reorder + (size_t)b * reorder_elems;

        for (size_t i = 0; i < a_elems; i++)
            batch_a[i] = (int8_t)((int)(i % 17) - 8);
        for (size_t i = 0; i < b_elems; i++)
            batch_b[i] = (int8_t)((int)(i % 13) - 6);

#if I8_BENCH_SVE_PACK
        pack_b_sve_i8(batch_b, batch_b_reo, K, N, ldb);
#else
        pack_b_neon_i8(batch_b, batch_b_reo, K, N, ldb);
#endif
        for (size_t copy = 0; copy < reorder_copies; copy++)
            pack_a8_i8(batch_a, batch_reorder + copy * (size_t)M * (size_t)K,
                       M, K, lda);
    }

    gemm_params_t params = {M, K, N, lda, K, ldc};

    for (int i = 0; i < warmup; i++) {
        for (int b = 0; b < batch_count; b++) {
            run_kernel_mt(A + (size_t)b * a_elems,
                          B_reo + (size_t)b * b_reo_bytes,
                          C + (size_t)b * c_elems,
                          A_reorder + (size_t)b * reorder_elems,
                          &params, threads);
        }
    }

    double best = 0.0;
    double ops = 2.0 * (double)M * (double)N * (double)K *
                 (double)batch_count;
    for (int r = 0; r < runs; r++) {
        double t0 = now_sec();
        for (int i = 0; i < reps; i++) {
            for (int b = 0; b < batch_count; b++) {
                run_kernel_mt(A + (size_t)b * a_elems,
                              B_reo + (size_t)b * b_reo_bytes,
                              C + (size_t)b * c_elems,
                              A_reorder + (size_t)b * reorder_elems,
                              &params, threads);
            }
        }
        double dt = (now_sec() - t0) / (double)reps;
        double gops = ops / dt / 1e9;
        if (gops > best)
            best = gops;
    }

    printf("%s,%s,%s,%d,%d,%d,%d,%.1f,%d,%.2f,%.2f,%.2f,%d,%d\n",
           variant, mode, cache_class(kib), M, K, N, threads, kib, reps,
           best, best * 100.0 / peak_gops,
           best * 100.0 / (peak_gops * (double)threads),
           stride_factor, batch_count);

    free(A);
    free(B);
    free(B_reo);
    free(A_reorder);
    free(C);
    return 0;
}
