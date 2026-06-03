#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>

typedef float    AccumType;
typedef uint16_t BF16Type;

void bf16gemm_k_ld(const BF16Type *A, const BF16Type *B_reordered,
                   AccumType *C, int m, int k, int n,
                   BF16Type *A_reorder, int lda, int ldb, int ldc);

static inline BF16Type float_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    uint32_t rounding = ((u >> 16) & 1) + 0x7FFF;
    u += rounding;
    return (BF16Type)(u >> 16);
}

static double get_time(struct timespec *s, struct timespec *e) {
    return (e->tv_sec - s->tv_sec) + (e->tv_nsec - s->tv_nsec) * 1e-9;
}

static void rand_bf16(BF16Type *p, int n) {
    for (int i = 0; i < n; i++) {
        float val = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        p[i] = float_to_bf16(val);
    }
}

static void reorder_B_bf16(const BF16Type* B, BF16Type* B_reo, int K, int N) {
    int idx = 0;
    for (int cb = 0; cb < N/8; ++cb) {
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
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s M K N [loops]\n", argv[0]);
        return 1;
    }
    int m = atoi(argv[1]), k = atoi(argv[2]), n = atoi(argv[3]);
    int loops = argc > 4 ? atoi(argv[4]) : 100;

    // Round up to multiples of 8 (M,N) and 8 (K)
    m = (m + 7) & ~7;
    k = (k + 7) & ~7;
    n = (n + 7) & ~7;

    srand(42);

    BF16Type *A = (BF16Type*)malloc(m * k * sizeof(BF16Type));
    BF16Type *B_orig = (BF16Type*)malloc(k * n * sizeof(BF16Type));
    BF16Type *B_reo = (BF16Type*)malloc(k * n * sizeof(BF16Type));
    AccumType *C = (AccumType*)calloc(m * n, sizeof(AccumType));
    BF16Type *A_reo = (BF16Type*)malloc(m * k * sizeof(BF16Type));

    rand_bf16(A, m * k);
    rand_bf16(B_orig, k * n);
    reorder_B_bf16(B_orig, B_reo, k, n);

    // Warm up
    for (int w = 0; w < 3; w++)
        bf16gemm_k_ld(A, B_reo, C, m, k, n, A_reo, k, k, n);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    for (int i = 0; i < loops; i++)
        bf16gemm_k_ld(A, B_reo, C, m, k, n, A_reo, k, k, n);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    double t = get_time(&start, &end);
    double flops = 2.0 * m * n * k;
    double gflops = flops / (t / loops) * 1e-9;
    double t_us = t * 1e6 / loops;

    printf("%5d %5d %5d  %8.3f  %7.2f\n", m, k, n, t_us, gflops);

    free(A); free(B_orig); free(B_reo); free(C); free(A_reo);
    return 0;
}
