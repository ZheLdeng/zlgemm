// bench_msplit.c -- compare the M-split/split-K lib vs the current M+N dispatch
// on the same pre-packed B (timing methodology matches bench_dispatch_types.c).
// Usage: bench_msplit M K N reps warmup runs threads
// Prints: M,K,N,threads,sve_gops,msplit_gops,ratio
#include <arm_sve.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "i8gemm.h"
#include "i8gemm_msplit.h"

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static int ru(int x, int q) { return ((x + q - 1) / q) * q; }
static int n_tile(void) { return (int)(svcntb() / 16) * 8; }
static void *xa(size_t n) { void *p = aligned_alloc(64, (n + 63) & ~(size_t)63); if (!p) exit(1); return p; }

int main(int argc, char **argv) {
    if (argc != 8) { fprintf(stderr, "usage: %s M K N reps warmup runs threads\n", argv[0]); return 2; }
    int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]);
    int reps = atoi(argv[4]), warmup = atoi(argv[5]), runs = atoi(argv[6]), t = atoi(argv[7]);
    int K_r = ru(K < 16 ? 16 : K, 16), N_r = ru(N < 8 ? 8 : N, n_tile());

    i8_t *A = xa((size_t)M * K_r); memset(A, 0, (size_t)M * K_r);
    i8_t *Bp = calloc((size_t)K_r * N_r, 1);
    i8_t *Br = xa((size_t)K_r * N_r);
    int32_t *C = xa((size_t)M * N_r * sizeof(int32_t));
    for (int i = 0; i < M; i++) for (int k = 0; k < K; k++) A[(size_t)i*K_r+k] = (i8_t)((i+k)%17-8);
    for (int k = 0; k < K; k++) for (int j = 0; j < N; j++) Bp[(size_t)k*N_r+j] = (i8_t)((k+j)%13-6);
    i8_pack_B(Bp, Br, K_r, N_r); free(Bp);

    const double ops = 2.0 * M * K * N;

    // current M+N dispatch
    for (int w = 0; w < warmup; w++) i8gemm_mt_dispatch(A, Br, C, M, K_r, N_r, t);
    double best_sve = 0;
    for (int r = 0; r < runs; r++) {
        double t0 = now_sec();
        for (int i = 0; i < reps; i++) i8gemm_mt_dispatch(A, Br, C, M, K_r, N_r, t);
        double g = ops / ((now_sec()-t0)/reps) / 1e9; if (g > best_sve) best_sve = g;
    }
    // msplit dispatch
    for (int w = 0; w < warmup; w++) i8gemm_msplit_dispatch(A, Br, C, M, K_r, N_r, t);
    double best_ms = 0;
    for (int r = 0; r < runs; r++) {
        double t0 = now_sec();
        for (int i = 0; i < reps; i++) i8gemm_msplit_dispatch(A, Br, C, M, K_r, N_r, t);
        double g = ops / ((now_sec()-t0)/reps) / 1e9; if (g > best_ms) best_ms = g;
    }
    printf("%d,%d,%d,%d,%.1f,%.1f,%.3f\n", M, K, N, t, best_sve, best_ms, best_ms/best_sve);
    free(A); free(Br); free(C);
    return 0;
}
