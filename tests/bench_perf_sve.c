// bench_perf_sve.c -- SVE peak and GEMM dispatch benchmark.

#include <arm_sve.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bf16gemm.h"
#include "i8gemm.h"

static double elapsed_sec(const struct timespec *s, const struct timespec *e) {
    return (e->tv_sec - s->tv_sec) + (e->tv_nsec - s->tv_nsec) * 1e-9;
}

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
    if (bytes == 0)
        bytes = 64;
    bytes = (bytes + 63u) & ~(size_t)63u;
    void *p = aligned_alloc(64, bytes);
    if (!p) {
        fprintf(stderr, "allocation failed: %zu bytes\n", bytes);
        exit(1);
    }
    return p;
}

static bf16_t float_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    u += ((u >> 16) & 1u) + 0x7fffu;
    return (bf16_t)(u >> 16);
}

static double measure_peak_sve_bfmmla(void) {
    volatile int64_t n;
    const int64_t iters = 10000000LL;
    const double ops_per_iter = 16.0 * 32.0 * (double)sve_segments();
    double best = 0.0;
    struct timespec s, e;

#define SVE_BFM_BODY \
    "bfmmla z0.s, z16.h, z24.h\n\t" \
    "bfmmla z1.s, z17.h, z25.h\n\t" \
    "bfmmla z2.s, z18.h, z26.h\n\t" \
    "bfmmla z3.s, z19.h, z27.h\n\t" \
    "bfmmla z4.s, z20.h, z28.h\n\t" \
    "bfmmla z5.s, z21.h, z29.h\n\t" \
    "bfmmla z6.s, z22.h, z30.h\n\t" \
    "bfmmla z7.s, z23.h, z31.h\n\t" \
    "bfmmla z8.s, z16.h, z24.h\n\t" \
    "bfmmla z9.s, z17.h, z25.h\n\t" \
    "bfmmla z10.s, z18.h, z26.h\n\t" \
    "bfmmla z11.s, z19.h, z27.h\n\t" \
    "bfmmla z12.s, z20.h, z28.h\n\t" \
    "bfmmla z13.s, z21.h, z29.h\n\t" \
    "bfmmla z14.s, z22.h, z30.h\n\t" \
    "bfmmla z15.s, z23.h, z31.h\n\t"

    for (int run = 0; run < 5; run++) {
        n = iters;
        clock_gettime(CLOCK_MONOTONIC_RAW, &s);
        __asm__ volatile(
            "dup z16.h, #1\n\t"
            "dup z17.h, #1\n\t"
            "dup z18.h, #1\n\t"
            "dup z19.h, #1\n\t"
            "dup z20.h, #1\n\t"
            "dup z21.h, #1\n\t"
            "dup z22.h, #1\n\t"
            "dup z23.h, #1\n\t"
            "dup z24.h, #1\n\t"
            "dup z25.h, #1\n\t"
            "dup z26.h, #1\n\t"
            "dup z27.h, #1\n\t"
            "dup z28.h, #1\n\t"
            "dup z29.h, #1\n\t"
            "dup z30.h, #1\n\t"
            "dup z31.h, #1\n\t"
            "mov z0.s, #0\n\t"
            "mov z1.s, #0\n\t"
            "mov z2.s, #0\n\t"
            "mov z3.s, #0\n\t"
            "mov z4.s, #0\n\t"
            "mov z5.s, #0\n\t"
            "mov z6.s, #0\n\t"
            "mov z7.s, #0\n\t"
            "mov z8.s, #0\n\t"
            "mov z9.s, #0\n\t"
            "mov z10.s, #0\n\t"
            "mov z11.s, #0\n\t"
            "mov z12.s, #0\n\t"
            "mov z13.s, #0\n\t"
            "mov z14.s, #0\n\t"
            "mov z15.s, #0\n\t"
            "1:\n\t"
            SVE_BFM_BODY
            "subs %[n], %[n], #1\n\t"
            "b.ne 1b\n\t"
            : [n] "+r"(n)
            :
            : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
              "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
              "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
              "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
              "cc", "memory");
        clock_gettime(CLOCK_MONOTONIC_RAW, &e);
        double sec = elapsed_sec(&s, &e);
        if (sec > 0.0) {
            double gflops = (double)iters * ops_per_iter / sec * 1e-9;
            if (gflops > best)
                best = gflops;
        }
    }
#undef SVE_BFM_BODY
    return best;
}

static double measure_peak_sve_smmla(void) {
    volatile int64_t n;
    const int64_t iters = 10000000LL;
    const double ops_per_iter = 16.0 * 64.0 * (double)sve_segments();
    double best = 0.0;
    struct timespec s, e;

#define SVE_SMM_BODY \
    "smmla z0.s, z16.b, z24.b\n\t" \
    "smmla z1.s, z17.b, z25.b\n\t" \
    "smmla z2.s, z18.b, z26.b\n\t" \
    "smmla z3.s, z19.b, z27.b\n\t" \
    "smmla z4.s, z20.b, z28.b\n\t" \
    "smmla z5.s, z21.b, z29.b\n\t" \
    "smmla z6.s, z22.b, z30.b\n\t" \
    "smmla z7.s, z23.b, z31.b\n\t" \
    "smmla z8.s, z16.b, z24.b\n\t" \
    "smmla z9.s, z17.b, z25.b\n\t" \
    "smmla z10.s, z18.b, z26.b\n\t" \
    "smmla z11.s, z19.b, z27.b\n\t" \
    "smmla z12.s, z20.b, z28.b\n\t" \
    "smmla z13.s, z21.b, z29.b\n\t" \
    "smmla z14.s, z22.b, z30.b\n\t" \
    "smmla z15.s, z23.b, z31.b\n\t"

    for (int run = 0; run < 5; run++) {
        n = iters;
        clock_gettime(CLOCK_MONOTONIC_RAW, &s);
        __asm__ volatile(
            "dup z16.b, #1\n\t"
            "dup z17.b, #1\n\t"
            "dup z18.b, #1\n\t"
            "dup z19.b, #1\n\t"
            "dup z20.b, #1\n\t"
            "dup z21.b, #1\n\t"
            "dup z22.b, #1\n\t"
            "dup z23.b, #1\n\t"
            "dup z24.b, #1\n\t"
            "dup z25.b, #1\n\t"
            "dup z26.b, #1\n\t"
            "dup z27.b, #1\n\t"
            "dup z28.b, #1\n\t"
            "dup z29.b, #1\n\t"
            "dup z30.b, #1\n\t"
            "dup z31.b, #1\n\t"
            "mov z0.s, #0\n\t"
            "mov z1.s, #0\n\t"
            "mov z2.s, #0\n\t"
            "mov z3.s, #0\n\t"
            "mov z4.s, #0\n\t"
            "mov z5.s, #0\n\t"
            "mov z6.s, #0\n\t"
            "mov z7.s, #0\n\t"
            "mov z8.s, #0\n\t"
            "mov z9.s, #0\n\t"
            "mov z10.s, #0\n\t"
            "mov z11.s, #0\n\t"
            "mov z12.s, #0\n\t"
            "mov z13.s, #0\n\t"
            "mov z14.s, #0\n\t"
            "mov z15.s, #0\n\t"
            "1:\n\t"
            SVE_SMM_BODY
            "subs %[n], %[n], #1\n\t"
            "b.ne 1b\n\t"
            : [n] "+r"(n)
            :
            : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
              "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
              "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
              "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
              "cc", "memory");
        clock_gettime(CLOCK_MONOTONIC_RAW, &e);
        double sec = elapsed_sec(&s, &e);
        if (sec > 0.0) {
            double gops = (double)iters * ops_per_iter / sec * 1e-9;
            if (gops > best)
                best = gops;
        }
    }
#undef SVE_SMM_BODY
    return best;
}

static void prepare_bf16(const bf16_t *A, const bf16_t *B,
                         bf16_t **A_pad, bf16_t **B_reo,
                         int M, int K, int N, int *K_r, int *N_r) {
    *K_r = round_up_int(K < 8 ? 8 : K, 8);
    *N_r = round_up_int(N < 8 ? 8 : N, sve_n_tile());
    *A_pad = (bf16_t *)xaligned_alloc((size_t)M * *K_r * sizeof(bf16_t));
    bf16_t *B_pad = (bf16_t *)calloc((size_t)*K_r * *N_r, sizeof(bf16_t));
    *B_reo = (bf16_t *)xaligned_alloc((size_t)*K_r * *N_r * sizeof(bf16_t));
    if (!B_pad)
        exit(1);
    memset(*A_pad, 0, (size_t)M * *K_r * sizeof(bf16_t));
    for (int i = 0; i < M; i++)
        memcpy(*A_pad + (size_t)i * *K_r, A + (size_t)i * K, (size_t)K * sizeof(bf16_t));
    for (int i = 0; i < K; i++)
        memcpy(B_pad + (size_t)i * *N_r, B + (size_t)i * N, (size_t)N * sizeof(bf16_t));
    bf16_pack_B(B_pad, *B_reo, *K_r, *N_r);
    free(B_pad);
}

static void prepare_i8(const int8_t *A, const int8_t *B,
                       int8_t **A_pad, int8_t **B_reo,
                       int M, int K, int N, int *K_r, int *N_r) {
    *K_r = round_up_int(K < 16 ? 16 : K, 16);
    *N_r = round_up_int(N < 8 ? 8 : N, sve_n_tile());
    *A_pad = (int8_t *)xaligned_alloc((size_t)M * *K_r);
    int8_t *B_pad = (int8_t *)calloc((size_t)*K_r * *N_r, 1);
    *B_reo = (int8_t *)xaligned_alloc((size_t)*K_r * *N_r);
    if (!B_pad)
        exit(1);
    memset(*A_pad, 0, (size_t)M * *K_r);
    for (int i = 0; i < M; i++)
        memcpy(*A_pad + (size_t)i * *K_r, A + (size_t)i * K, (size_t)K);
    for (int i = 0; i < K; i++)
        memcpy(B_pad + (size_t)i * *N_r, B + (size_t)i * N, (size_t)N);
    i8_pack_B(B_pad, *B_reo, *K_r, *N_r);
    free(B_pad);
}

static double bench_bf16_dispatch(int M, int K, int N, int nth) {
    bf16_t *A = (bf16_t *)xaligned_alloc((size_t)M * K * sizeof(bf16_t));
    bf16_t *B = (bf16_t *)xaligned_alloc((size_t)K * N * sizeof(bf16_t));
    bf16_t *A_pad, *B_reo;
    int K_r, N_r;
    for (int i = 0; i < M * K; i++)
        A[i] = float_to_bf16((float)((i % 17) - 8) * 0.125f);
    for (int i = 0; i < K * N; i++)
        B[i] = float_to_bf16((float)((i % 13) - 6) * 0.125f);
    prepare_bf16(A, B, &A_pad, &B_reo, M, K, N, &K_r, &N_r);
    float *C = (float *)xaligned_alloc((size_t)M * N_r * sizeof(float));

    bf16gemm_mt_dispatch(A_pad, B_reo, C, M, K_r, N_r, nth);
    double best = 0.0;
    for (int run = 0; run < 5; run++) {
        memset(C, 0, (size_t)M * N_r * sizeof(float));
        struct timespec s, e;
        clock_gettime(CLOCK_MONOTONIC_RAW, &s);
        bf16gemm_mt_dispatch(A_pad, B_reo, C, M, K_r, N_r, nth);
        clock_gettime(CLOCK_MONOTONIC_RAW, &e);
        double sec = elapsed_sec(&s, &e);
        if (sec > 0.0) {
            double gflops = 2.0 * (double)M * (double)K * (double)N / sec * 1e-9;
            if (gflops > best)
                best = gflops;
        }
    }
    free(A);
    free(B);
    free(A_pad);
    free(B_reo);
    free(C);
    return best;
}

static double bench_i8_dispatch(int M, int K, int N, int nth) {
    int8_t *A = (int8_t *)xaligned_alloc((size_t)M * K);
    int8_t *B = (int8_t *)xaligned_alloc((size_t)K * N);
    int8_t *A_pad, *B_reo;
    int K_r, N_r;
    for (int i = 0; i < M * K; i++)
        A[i] = (int8_t)((i % 17) - 8);
    for (int i = 0; i < K * N; i++)
        B[i] = (int8_t)((i % 13) - 6);
    prepare_i8(A, B, &A_pad, &B_reo, M, K, N, &K_r, &N_r);
    int32_t *C = (int32_t *)xaligned_alloc((size_t)M * N_r * sizeof(int32_t));

    i8gemm_mt_dispatch(A_pad, B_reo, C, M, K_r, N_r, nth);
    double best = 0.0;
    for (int run = 0; run < 5; run++) {
        memset(C, 0, (size_t)M * N_r * sizeof(int32_t));
        struct timespec s, e;
        clock_gettime(CLOCK_MONOTONIC_RAW, &s);
        i8gemm_mt_dispatch(A_pad, B_reo, C, M, K_r, N_r, nth);
        clock_gettime(CLOCK_MONOTONIC_RAW, &e);
        double sec = elapsed_sec(&s, &e);
        if (sec > 0.0) {
            double gops = 2.0 * (double)M * (double)K * (double)N / sec * 1e-9;
            if (gops > best)
                best = gops;
        }
    }
    free(A);
    free(B);
    free(A_pad);
    free(B_reo);
    free(C);
    return best;
}

static int parse_shape_csv_line(const char *line, int *M, int *K, int *N) {
    int csv_m, csv_n, csv_k;
    if (sscanf(line, " %d , %d , %d", &csv_m, &csv_n, &csv_k) != 3)
        return 0;
    if (csv_m <= 0 || csv_n <= 0 || csv_k <= 0)
        return 0;
    *M = csv_m;
    *K = csv_k;
    *N = csv_n;
    return 1;
}

static void print_result(int M, int K, int N, int nth,
                         double peak_bf16, double peak_i8,
                         double bf16_gflops, double i8_gops) {
    double bf16_eff = peak_bf16 > 0.0 ? bf16_gflops / (peak_bf16 * nth) * 100.0 : 0.0;
    double i8_eff = peak_i8 > 0.0 ? i8_gops / (peak_i8 * nth) * 100.0 : 0.0;
    printf("%d,%d,%d,%d,%.2f,%.2f,%.2f,%.2f\n",
           M, K, N, nth, bf16_gflops, bf16_eff, i8_gops, i8_eff);
}

static void run_one(int M, int K, int N, int nth, double peak_bf16, double peak_i8) {
    double bf16_gflops = bench_bf16_dispatch(M, K, N, nth);
    double i8_gops = bench_i8_dispatch(M, K, N, nth);
    print_result(M, K, N, nth, peak_bf16, peak_i8, bf16_gflops, i8_gops);
}

static void run_csv(const char *path, double peak_bf16, double peak_i8) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        perror(path);
        exit(1);
    }
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        int M, K, N;
        if (!parse_shape_csv_line(line, &M, &K, &N))
            continue;
        run_one(M, K, N, 1, peak_bf16, peak_i8);
    }
    fclose(fp);
}

static void run_sweep(int M, int K, int N, double peak_bf16, double peak_i8) {
    const int threads[] = {1, 2, 4, 8, 16, 32, 48, 64, 80};
    int max_threads = omp_get_num_procs();
    for (size_t i = 0; i < sizeof(threads) / sizeof(threads[0]); i++) {
        if (threads[i] <= max_threads)
            run_one(M, K, N, threads[i], peak_bf16, peak_i8);
    }
}

int main(int argc, char **argv) {
    double peak_bf16 = measure_peak_sve_bfmmla();
    double peak_i8 = measure_peak_sve_smmla();
    printf("SVE VL=%zu bits, 128b segments=%d, N tile=%d\n",
           svcntb() * 8, sve_segments(), sve_n_tile());
    printf("SVE peak: bf16 %.2f GFLOPS/thread, i8 %.2f GOPS/thread\n",
           peak_bf16, peak_i8);
    printf("M,K,N,threads,bf16_GFLOPS,bf16_eff_pct,i8_GOPS,i8_eff_pct\n");

    if (argc == 1) {
        run_csv("shape.csv", peak_bf16, peak_i8);
    } else if (argc == 2) {
        run_csv(argv[1], peak_bf16, peak_i8);
    } else if (argc == 5 && strcmp(argv[1], "--mt-sweep-both") == 0) {
        run_sweep(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), peak_bf16, peak_i8);
    } else if ((argc == 5 || argc == 6) && strcmp(argv[1], "--mt-both") == 0) {
        int nth = argc == 6 ? atoi(argv[5]) : 1;
        run_one(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), nth, peak_bf16, peak_i8);
    } else {
        fprintf(stderr, "usage: %s [shape.csv]\n", argv[0]);
        fprintf(stderr, "       %s --mt-both M K N [threads]\n", argv[0]);
        fprintf(stderr, "       %s --mt-sweep-both M K N\n", argv[0]);
        return 1;
    }
    return 0;
}
