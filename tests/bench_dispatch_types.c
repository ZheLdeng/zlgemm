// bench_dispatch_types.c -- benchmark BF16/I8 public dispatch paths.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef BENCH_SVE
#include <arm_sve.h>
#endif

#include "bf16gemm.h"
#include "i8gemm.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int round_up_int(int x, int q) {
    return ((x + q - 1) / q) * q;
}

static int n_tile(void) {
#ifdef BENCH_SVE
    return (int)(svcntb() / 16) * 8;
#else
    return 8;
#endif
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

static bf16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    u += ((u >> 16) & 1u) + 0x7fffu;
    return (bf16_t)(u >> 16);
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

static double env_double(const char *name, double fallback) {
    const char *s = getenv(name);
    if (!s || !*s)
        return fallback;
    char *end = NULL;
    double v = strtod(s, &end);
    return end != s && v > 0.0 ? v : fallback;
}

static void prepare_bf16(const bf16_t *A, const bf16_t *B,
                         bf16_t **A_pad, bf16_t **B_reo,
                         int M, int K, int N, int *K_r, int *N_r) {
    *K_r = round_up_int(K < 8 ? 8 : K, 8);
    *N_r = round_up_int(N < 8 ? 8 : N, n_tile());
    *A_pad = (bf16_t *)xalloc((size_t)M * (size_t)*K_r * sizeof(bf16_t));
    bf16_t *B_pad = (bf16_t *)calloc((size_t)*K_r * (size_t)*N_r,
                                     sizeof(bf16_t));
    *B_reo = (bf16_t *)xalloc((size_t)*K_r * (size_t)*N_r * sizeof(bf16_t));
    if (!B_pad)
        exit(1);
    memset(*A_pad, 0, (size_t)M * (size_t)*K_r * sizeof(bf16_t));
    for (int i = 0; i < M; i++)
        memcpy(*A_pad + (size_t)i * (size_t)*K_r,
               A + (size_t)i * (size_t)K, (size_t)K * sizeof(bf16_t));
    for (int i = 0; i < K; i++)
        memcpy(B_pad + (size_t)i * (size_t)*N_r,
               B + (size_t)i * (size_t)N, (size_t)N * sizeof(bf16_t));
    bf16_pack_B(B_pad, *B_reo, *K_r, *N_r);
    free(B_pad);
}

static void prepare_i8(const i8_t *A, const i8_t *B,
                       i8_t **A_pad, i8_t **B_reo,
                       int M, int K, int N, int *K_r, int *N_r) {
    *K_r = round_up_int(K < 16 ? 16 : K, 16);
    *N_r = round_up_int(N < 8 ? 8 : N, n_tile());
    *A_pad = (i8_t *)xalloc((size_t)M * (size_t)*K_r);
    i8_t *B_pad = (i8_t *)calloc((size_t)*K_r * (size_t)*N_r, 1);
    *B_reo = (i8_t *)xalloc((size_t)*K_r * (size_t)*N_r);
    if (!B_pad)
        exit(1);
    memset(*A_pad, 0, (size_t)M * (size_t)*K_r);
    for (int i = 0; i < M; i++)
        memcpy(*A_pad + (size_t)i * (size_t)*K_r,
               A + (size_t)i * (size_t)K, (size_t)K);
    for (int i = 0; i < K; i++)
        memcpy(B_pad + (size_t)i * (size_t)*N_r,
               B + (size_t)i * (size_t)N, (size_t)N);
    i8_pack_B(B_pad, *B_reo, *K_r, *N_r);
    free(B_pad);
}

static double bench_bf16_one(int M, int K, int N, int reps, int warmup,
                             int runs, int threads, int batch_count,
                             double *kib_out) {
    int K_r = 0, N_r = 0;
    bf16_t **A_pad = (bf16_t **)xalloc((size_t)batch_count * sizeof(*A_pad));
    bf16_t **B_reo = (bf16_t **)xalloc((size_t)batch_count * sizeof(*B_reo));
    float **C = (float **)xalloc((size_t)batch_count * sizeof(*C));

    for (int b = 0; b < batch_count; b++) {
        bf16_t *A = (bf16_t *)xalloc((size_t)M * (size_t)K * sizeof(*A));
        bf16_t *B = (bf16_t *)xalloc((size_t)K * (size_t)N * sizeof(*B));
        for (int i = 0; i < M * K; i++)
            A[i] = f32_to_bf16((float)(((i + b) % 17) - 8) * 0.125f);
        for (int i = 0; i < K * N; i++)
            B[i] = f32_to_bf16((float)(((i + b) % 13) - 6) * 0.125f);
        prepare_bf16(A, B, &A_pad[b], &B_reo[b], M, K, N, &K_r, &N_r);
        C[b] = (float *)xalloc((size_t)M * (size_t)N_r * sizeof(float));
        free(A);
        free(B);
    }

    *kib_out = (double)((size_t)batch_count *
                        ((size_t)M * (size_t)K_r * sizeof(bf16_t) +
                         (size_t)K_r * (size_t)N_r * sizeof(bf16_t) +
                         (size_t)M * (size_t)N_r * sizeof(float))) / 1024.0;

    for (int w = 0; w < warmup; w++)
        for (int b = 0; b < batch_count; b++)
            bf16gemm_mt_dispatch(A_pad[b], B_reo[b], C[b], M, K_r, N_r,
                                 threads);

    double best = 0.0;
    const double ops = 2.0 * (double)M * (double)K * (double)N *
                       (double)batch_count;
    for (int r = 0; r < runs; r++) {
        double t0 = now_sec();
        for (int i = 0; i < reps; i++)
            for (int b = 0; b < batch_count; b++)
                bf16gemm_mt_dispatch(A_pad[b], B_reo[b], C[b], M, K_r, N_r,
                                     threads);
        double dt = (now_sec() - t0) / (double)reps;
        double gflops = ops / dt / 1e9;
        if (gflops > best)
            best = gflops;
    }

    for (int b = 0; b < batch_count; b++) {
        free(A_pad[b]);
        free(B_reo[b]);
        free(C[b]);
    }
    free(A_pad);
    free(B_reo);
    free(C);
    return best;
}

static double bench_i8_one(int M, int K, int N, int reps, int warmup,
                           int runs, int threads, int batch_count,
                           double *kib_out) {
    int K_r = 0, N_r = 0;
    i8_t **A_pad = (i8_t **)xalloc((size_t)batch_count * sizeof(*A_pad));
    i8_t **B_reo = (i8_t **)xalloc((size_t)batch_count * sizeof(*B_reo));
    i32_t **C = (i32_t **)xalloc((size_t)batch_count * sizeof(*C));

    for (int b = 0; b < batch_count; b++) {
        i8_t *A = (i8_t *)xalloc((size_t)M * (size_t)K);
        i8_t *B = (i8_t *)xalloc((size_t)K * (size_t)N);
        for (int i = 0; i < M * K; i++)
            A[i] = (i8_t)(((i + b) % 17) - 8);
        for (int i = 0; i < K * N; i++)
            B[i] = (i8_t)(((i + b) % 13) - 6);
        prepare_i8(A, B, &A_pad[b], &B_reo[b], M, K, N, &K_r, &N_r);
        C[b] = (i32_t *)xalloc((size_t)M * (size_t)N_r * sizeof(i32_t));
        free(A);
        free(B);
    }

    *kib_out = (double)((size_t)batch_count *
                        ((size_t)M * (size_t)K_r +
                         (size_t)K_r * (size_t)N_r +
                         (size_t)M * (size_t)N_r * sizeof(i32_t))) / 1024.0;

    for (int w = 0; w < warmup; w++)
        for (int b = 0; b < batch_count; b++)
            i8gemm_mt_dispatch(A_pad[b], B_reo[b], C[b], M, K_r, N_r,
                               threads);

    double best = 0.0;
    const double ops = 2.0 * (double)M * (double)K * (double)N *
                       (double)batch_count;
    for (int r = 0; r < runs; r++) {
        double t0 = now_sec();
        for (int i = 0; i < reps; i++)
            for (int b = 0; b < batch_count; b++)
                i8gemm_mt_dispatch(A_pad[b], B_reo[b], C[b], M, K_r, N_r,
                                   threads);
        double dt = (now_sec() - t0) / (double)reps;
        double gops = ops / dt / 1e9;
        if (gops > best)
            best = gops;
    }

    for (int b = 0; b < batch_count; b++) {
        free(A_pad[b]);
        free(B_reo[b]);
        free(C[b]);
    }
    free(A_pad);
    free(B_reo);
    free(C);
    return best;
}

static void print_row(const char *impl, const char *dtype, int M, int K, int N,
                      int threads, int batch_count, double kib, int reps,
                      double perf, double baseline) {
    printf("%s,%s,%s,%d,%d,%d,%d,%d,%.1f,%d,%.2f,%.2f,%.2f\n",
           impl, dtype, cache_class(kib), M, K, N, threads, batch_count, kib,
           reps, perf, perf * 100.0 / baseline,
           perf * 100.0 / (baseline * (double)threads));
}

int main(int argc, char **argv) {
    if (argc != 10 && argc != 11) {
        fprintf(stderr,
                "usage: %s impl bf16|i8|both M K N reps warmup runs threads [batch_count]\n",
                argv[0]);
        return 2;
    }
    const char *impl = argv[1];
    const char *dtype = argv[2];
    int M = atoi(argv[3]);
    int K = atoi(argv[4]);
    int N = atoi(argv[5]);
    int reps = atoi(argv[6]);
    int warmup = atoi(argv[7]);
    int runs = atoi(argv[8]);
    int threads = atoi(argv[9]);
    int batch_count = argc == 11 ? atoi(argv[10]) : 1;
    if (threads < 1 || batch_count < 1)
        return 2;

    const double bf16_base = env_double("BF16_BASELINE_GFLOPS", 330.0);
    const double i8_base = env_double("I8_BASELINE_GOPS", 660.0);
    double kib = 0.0;

    if (strcmp(dtype, "bf16") == 0 || strcmp(dtype, "both") == 0) {
        double perf = bench_bf16_one(M, K, N, reps, warmup, runs, threads,
                                     batch_count, &kib);
        print_row(impl, "bf16", M, K, N, threads, batch_count, kib, reps,
                  perf, bf16_base);
    }
    if (strcmp(dtype, "i8") == 0 || strcmp(dtype, "both") == 0) {
        double perf = bench_i8_one(M, K, N, reps, warmup, runs, threads,
                                   batch_count, &kib);
        print_row(impl, "i8", M, K, N, threads, batch_count, kib, reps,
                  perf, i8_base);
    }
    return 0;
}
