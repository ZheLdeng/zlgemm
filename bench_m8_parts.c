// bench_m8_parts.c -- microbenchmark one M8 BF16 kernel variant.
//
// The shell driver builds several assembly variants that export the same
// three M8 no-C-load entry points:
//   bf16gemm_k_nld_f_m8       fp32 output
//   bf16gemm_k_nld_b_m8       bf16 output
//   bf16gemm_k_nld_bias_f_m8  fp32 output with fp32 bias

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

typedef struct {
    int m;
    int k;
    int n;
    int lda;
    int ldb;
    int ldc;
} gemm_params_t;

extern void bf16gemm_k_nld_f_m8(const uint16_t *A, const uint16_t *B_reo,
                                float *C, uint16_t *A_reorder,
                                const gemm_params_t *params);
extern void bf16gemm_k_nld_b_m8(const uint16_t *A, const uint16_t *B_reo,
                                uint16_t *C, uint16_t *A_reorder,
                                const gemm_params_t *params);
extern void bf16gemm_k_nld_bias_f_m8(const uint16_t *A, const uint16_t *B_reo,
                                     float *C, uint16_t *A_reorder,
                                     const gemm_params_t *params,
                                     const float *bias);

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

static void fill_bias(float *bias, int n) {
    for (int i = 0; i < n; i++)
        bias[i] = (float)((i % 17) - 8) * 0.0625f;
}

static void pack_a8(const uint16_t *A, uint16_t *P, int M, int K) {
    size_t idx = 0;
    for (int mb = 0; mb < M; mb += 8) {
        for (int kb = 0; kb < K; kb += 4) {
            for (int rp = 0; rp < 4; rp++) {
                int r0 = mb + rp * 2;
                int r1 = r0 + 1;
                for (int k = 0; k < 4; k++)
                    P[idx++] = A[(size_t)r0 * K + kb + k];
                for (int k = 0; k < 4; k++)
                    P[idx++] = A[(size_t)r1 * K + kb + k];
            }
        }
    }
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

static int mode_is_bf16(const char *mode) {
    return strcmp(mode, "bf16") == 0;
}

static int mode_is_bias(const char *mode) {
    return strcmp(mode, "bias") == 0;
}

static void run_kernel(const char *mode, const uint16_t *A, const uint16_t *B,
                       float *C_f32, uint16_t *C_bf16, uint16_t *A_reorder,
                       const gemm_params_t *params, const float *bias) {
    if (mode_is_bf16(mode)) {
        bf16gemm_k_nld_b_m8(A, B, C_bf16, A_reorder, params);
    } else if (mode_is_bias(mode)) {
        bf16gemm_k_nld_bias_f_m8(A, B, C_f32, A_reorder, params, bias);
    } else {
        bf16gemm_k_nld_f_m8(A, B, C_f32, A_reorder, params);
    }
}

static void run_kernel_mt(const char *mode, const uint16_t *A, const uint16_t *B,
                          float *C_f32, uint16_t *C_bf16, uint16_t *A_reorder,
                          const gemm_params_t *params, const float *bias,
                          int threads) {
    if (threads <= 1) {
        run_kernel(mode, A, B, C_f32, C_bf16, A_reorder, params, bias);
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
            run_kernel(mode,
                       A + (size_t)m_start * (size_t)params->lda,
                       B,
                       C_f32 + (size_t)m_start * (size_t)params->ldc,
                       C_bf16 + (size_t)m_start * (size_t)params->ldc,
                       A_reorder + (size_t)m_start * (size_t)params->k,
                       &local,
                       bias);
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 8 && argc != 9 && argc != 10) {
        fprintf(stderr,
                "usage: %s variant M K N reps warmup runs [f32|bf16|bias] [threads]\n",
                argv[0]);
        return 2;
    }

    const char *variant = argv[1];
    int M = atoi(argv[2]);
    int K = atoi(argv[3]);
    int N = atoi(argv[4]);
    int reps = atoi(argv[5]);
    int warmup = atoi(argv[6]);
    int runs = atoi(argv[7]);
    const char *mode = argc >= 9 ? argv[8] : "f32";
    int threads = argc == 10 ? atoi(argv[9]) : 1;
    if (strcmp(mode, "f32") != 0 && strcmp(mode, "bf16") != 0 &&
        strcmp(mode, "bias") != 0) {
        fprintf(stderr, "bad mode: %s\n", mode);
        return 2;
    }
    if (threads < 1) {
        fprintf(stderr, "bad threads: %d\n", threads);
        return 2;
    }
    if (M % 8 != 0) {
        fprintf(stderr, "M must be a multiple of 8 for this M8 bench: %d\n", M);
        return 2;
    }

    const double peak_gflops = 330.0;
    size_t a_bytes = (size_t)M * (size_t)K * sizeof(uint16_t);
    size_t b_bytes = (size_t)K * (size_t)N * sizeof(uint16_t);
    size_t c_bytes = (size_t)M * (size_t)N *
                     (mode_is_bf16(mode) ? sizeof(uint16_t) : sizeof(float));
    size_t bias_bytes = mode_is_bias(mode) ? (size_t)N * sizeof(float) : 0;
    double kib = (double)(a_bytes + b_bytes + c_bytes + bias_bytes) / 1024.0;

    uint16_t *A = (uint16_t *)xalloc(a_bytes);
    uint16_t *B = (uint16_t *)xalloc(b_bytes);
    uint16_t *A_reorder = (uint16_t *)xalloc(a_bytes);
    float *C_f32 = (float *)xalloc((size_t)M * (size_t)N * sizeof(float));
    uint16_t *C_bf16 = (uint16_t *)xalloc((size_t)M * (size_t)N * sizeof(uint16_t));
    float *bias = (float *)xalloc((size_t)N * sizeof(float));

    pack_a8(A, A_reorder, M, K);
    fill_bias(bias, N);
    gemm_params_t params = {M, K, N, K, K, N};

    for (int i = 0; i < warmup; i++)
        run_kernel_mt(mode, A, B, C_f32, C_bf16, A_reorder, &params, bias,
                      threads);

    double best = 0.0;
    double ops = 2.0 * (double)M * (double)N * (double)K;
    for (int r = 0; r < runs; r++) {
        double t0 = now_sec();
        for (int i = 0; i < reps; i++)
            run_kernel_mt(mode, A, B, C_f32, C_bf16, A_reorder, &params, bias,
                          threads);
        double dt = (now_sec() - t0) / (double)reps;
        double gflops = ops / dt / 1e9;
        if (gflops > best)
            best = gflops;
    }

    printf("%s,%s,%s,%d,%d,%d,%d,%.1f,%d,%.2f,%.2f,%.2f\n",
           variant, mode, cache_class(kib), M, K, N, threads, kib, reps,
           best, best * 100.0 / peak_gflops,
           best * 100.0 / (peak_gflops * (double)threads));

    free(A);
    free(B);
    free(A_reorder);
    free(C_f32);
    free(C_bf16);
    free(bias);
    return 0;
}
