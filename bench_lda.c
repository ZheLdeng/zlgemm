#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <assert.h>

typedef int32_t MatrixType;
typedef int8_t  DFMatrixType;
typedef int8_t  BMatrixType;

// i8gemm_k_ldc:  A 外部重排
void i8gemm_k_ldc(const DFMatrixType *A_reordered, const BMatrixType *B_reordered,
                  MatrixType *C, int m, int k, int n);
// i8gemm_k_lda: A 内部重排
void i8gemm_k_lda(const DFMatrixType *A, const BMatrixType *B_reordered,
                  MatrixType *C, int m, int k, int n, int8_t *A_reorder);
// i8gemm_k_pack: A + B 均在内部重排
void i8gemm_k_pack(const DFMatrixType *A, const BMatrixType *B_orig,
                   MatrixType *C, int m, int k, int n,
                   int8_t *A_reorder, int8_t *B_reorder);

static double get_time(struct timespec *s, struct timespec *e) {
    return (e->tv_sec - s->tv_sec) + (e->tv_nsec - s->tv_nsec) * 1e-9;
}

static void rand_i8(int8_t *p, int n) {
    for (int i = 0; i < n; i++) p[i] = (int8_t)(rand() % 256 - 128);
}

static void reorder_A_1x8(const int8_t* A, int8_t* A_reordered, int M, int K) {
    int idx = 0;
    for (int rb = 0; rb < M/8; ++rb)
        for (int cb = 0; cb < K/8; ++cb)
            for (int i = 0; i < 8; ++i)
                for (int j = 0; j < 8; ++j)
                    A_reordered[idx++] = A[(rb*8 + i) * K + (cb*8 + j)];
}

static void reorder_B_8x1(const int8_t* B, int8_t* B_reordered, int K, int N) {
    int idx = 0;
    for (int cb = 0; cb < N/8; ++cb)
        for (int rb = 0; rb < K/8; ++rb)
            for (int j = 0; j < 8; ++j)
                for (int i = 0; i < 8; ++i)
                    B_reordered[idx++] = B[(rb*8 + i) * N + (cb*8 + j)];
}

/* warm-up + timed loop helper */
static double bench_ldc(int8_t *A, int8_t *B_reo, int32_t *C,
                        int m, int k, int n, int loops) {
    int8_t *A_reo = malloc(m * k);
    struct timespec s, e;

    reorder_A_1x8(A, A_reo, m, k);                  // pre-reorder (once)
    for (int w = 0; w < 3; w++)
        i8gemm_k_ldc(A_reo, B_reo, C, m, k, n);  // warm

    clock_gettime(CLOCK_MONOTONIC_RAW, &s);
    for (int i = 0; i < loops; i++)
        i8gemm_k_ldc(A_reo, B_reo, C, m, k, n);
    clock_gettime(CLOCK_MONOTONIC_RAW, &e);

    free(A_reo);
    return get_time(&s, &e);
}

static double bench_lda(int8_t *A, int8_t *B_reo, int32_t *C,
                        int m, int k, int n, int loops) {
    int8_t *A_reo = malloc(m * k);  // buffer, filled by kernel
    struct timespec s, e;

    for (int w = 0; w < 3; w++)
        i8gemm_k_lda(A, B_reo, C, m, k, n, A_reo);  // warm

    clock_gettime(CLOCK_MONOTONIC_RAW, &s);
    for (int i = 0; i < loops; i++)
        i8gemm_k_lda(A, B_reo, C, m, k, n, A_reo);
    clock_gettime(CLOCK_MONOTONIC_RAW, &e);

    free(A_reo);
    return get_time(&s, &e);
}

static double bench_pack(int8_t *A, int8_t *B_orig, int32_t *C,
                         int m, int k, int n, int loops) {
    int8_t *A_reo = malloc(m * k);
    int8_t *B_reo = malloc(k * n);
    struct timespec s, e;

    for (int w = 0; w < 3; w++)
        i8gemm_k_pack(A, B_orig, C, m, k, n, A_reo, B_reo);  // warm

    clock_gettime(CLOCK_MONOTONIC_RAW, &s);
    for (int i = 0; i < loops; i++)
        i8gemm_k_pack(A, B_orig, C, m, k, n, A_reo, B_reo);
    clock_gettime(CLOCK_MONOTONIC_RAW, &e);

    free(A_reo); free(B_reo);
    return get_time(&s, &e);
}

int main(int argc, char *argv[]) {
    if (argc < 4) { fprintf(stderr, "Usage: %s M K N [loops]\n", argv[0]); return 1; }
    int m = atoi(argv[1]), k = atoi(argv[2]), n = atoi(argv[3]);
    int loops = argc > 4 ? atoi(argv[4]) : 100;

    srand(42);  // fixed seed for reproducibility

    int8_t *A = malloc(m * k), *B = malloc(k * n);
    int8_t *B_reo = malloc(k * n);
    int32_t *C = calloc(m * n, sizeof(int32_t));

    rand_i8(A, m * k); rand_i8(B, k * n);
    reorder_B_8x1(B, B_reo, k, n);

    /* ---- verify correctness once ---- */
    {
        int8_t *A_reo = malloc(m * k);
        int32_t *C_ldc = calloc(m * n, sizeof(int32_t));
        int32_t *C_lda = calloc(m * n, sizeof(int32_t));
        int32_t *C_pack = calloc(m * n, sizeof(int32_t));
        int8_t *A_buf = malloc(m * k);
        int8_t *B_buf = malloc(k * n);

        reorder_A_1x8(A, A_reo, m, k);
        i8gemm_k_ldc(A_reo, B_reo, C_ldc, m, k, n);
        i8gemm_k_lda(A, B_reo, C_lda, m, k, n, A_buf);
        i8gemm_k_pack(A, B, C_pack, m, k, n, A_buf, B_buf);

        // verify A_buf == A_reo (lda's internal reorder matches C reorder)
        int reo_ok = !memcmp(A_reo, A_buf, m * k);
        // verify ldc == lda
        int cmp_ldc_lda = !memcmp(C_ldc, C_lda, m * n * sizeof(int32_t));
        // verify ldc == pack
        int cmp_ldc_pack = !memcmp(C_ldc, C_pack, m * n * sizeof(int32_t));
        // verify pack B output matches C reorder
        int pack_b_ok = !memcmp(B_reo, B_buf, k * n);

        if (!reo_ok || !cmp_ldc_lda || !cmp_ldc_pack || !pack_b_ok) {
            printf("# VERIFY FAIL: A_reorder=%d lda_vs_ldc=%d pack_vs_ldc=%d B_reorder=%d\n",
                   reo_ok, cmp_ldc_lda, cmp_ldc_pack, pack_b_ok);
        } else {
            printf("# VERIFY OK: all kernels match\n");
        }
        free(A_reo); free(C_ldc); free(C_lda); free(C_pack);
        free(A_buf); free(B_buf);
    }

    double t_ldc = bench_ldc(A, B_reo, C, m, k, n, loops);
    double t_lda = bench_lda(A, B_reo, C, m, k, n, loops);
    double t_pack = bench_pack(A, B, C, m, k, n, loops);

    double flops = 2.0 * m * n * k;
    double gflops_ldc = flops / (t_ldc / loops) * 1e-9;
    double gflops_lda = flops / (t_lda / loops) * 1e-9;
    double gflops_pack = flops / (t_pack / loops) * 1e-9;

    printf("%5d %5d %5d  %8.3f  %8.3f  %8.3f  %7.2f  %7.2f  %7.2f  %+6.1f%%  %+6.1f%%\n",
           m, k, n,
           t_ldc * 1e6 / loops,
           t_lda * 1e6 / loops,
           t_pack * 1e6 / loops,
           gflops_ldc, gflops_lda, gflops_pack,
           (gflops_lda / gflops_ldc - 1.0) * 100,
           (gflops_pack / gflops_ldc - 1.0) * 100);

    free(A); free(B); free(B_reo); free(C);
    return 0;
}
