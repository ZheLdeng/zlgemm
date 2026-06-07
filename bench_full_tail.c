// bench_full_tail.c -- benchmark full BF16 dispatch for M-tail impact.
//
// This intentionally uses the public full dispatch path, not the standalone
// M8 attribution kernel, so arbitrary M such as 17/18/20 is safe.

#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

#include "bf16gemm.h"

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

static int round_up_int(int x, int q) {
    return ((x + q - 1) / q) * q;
}

static int default_n_align(void) {
#if defined(__ARM_FEATURE_SVE)
    int n = (int)(svcntb() / 2);
    return n > 8 ? n : 8;
#else
    return 8;
#endif
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

int main(int argc, char **argv) {
    if (argc != 9 && argc != 10) {
        fprintf(stderr,
                "usage: %s variant M K N reps warmup runs threads [n_align]\n",
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
    int threads = atoi(argv[8]);
    int n_align = argc == 10 ? atoi(argv[9]) : default_n_align();
    if (n_align <= 0)
        n_align = default_n_align();
    if (M <= 0 || K <= 0 || N <= 0 || reps <= 0 || warmup < 0 ||
        runs <= 0 || threads <= 0) {
        fprintf(stderr, "bad argument\n");
        return 2;
    }
    if (n_align < 8)
        n_align = 8;

    int K_r = round_up_int(K, 8);
    int N_r = round_up_int(N, n_align);

    size_t a_bytes = (size_t)M * (size_t)K_r * sizeof(bf16_t);
    size_t b_bytes = (size_t)K_r * (size_t)N_r * sizeof(bf16_t);
    size_t c_bytes = (size_t)M * (size_t)N_r * sizeof(f32_t);
    double kib = (double)(a_bytes + b_bytes + c_bytes) / 1024.0;

    bf16_t *A = (bf16_t *)xalloc(a_bytes);
    bf16_t *B = (bf16_t *)xalloc(b_bytes);
    bf16_t *B_reo = (bf16_t *)xalloc(b_bytes);
    f32_t *C = (f32_t *)xalloc(c_bytes);

    bf16_pack_B(B, B_reo, K_r, N_r);

    for (int i = 0; i < warmup; i++)
        bf16gemm_mt_dispatch(A, B_reo, C, M, K_r, N_r, threads);

    double best = 0.0;
    double ops = 2.0 * (double)M * (double)K_r * (double)N_r;
    for (int r = 0; r < runs; r++) {
        double t0 = now_sec();
        for (int i = 0; i < reps; i++)
            bf16gemm_mt_dispatch(A, B_reo, C, M, K_r, N_r, threads);
        double dt = (now_sec() - t0) / (double)reps;
        double gflops = ops / dt / 1e9;
        if (gflops > best)
            best = gflops;
    }

    const double peak_gflops = 330.0;
    printf("%s,full,%s,%d,%d,%d,%d,%.1f,%d,%.2f,%.2f,%.2f\n",
           variant, cache_class(kib), M, K_r, N_r, threads, kib, reps, best,
           best * 100.0 / peak_gflops,
           best * 100.0 / (peak_gflops * (double)threads));

    free(A);
    free(B);
    free(B_reo);
    free(C);
    return 0;
}
