// prepacked_ceiling.c -- measure the real kernels with data PRE-PACKED and
// cache-resident, so memory layout/repack and DRAM bandwidth are excluded.
// This isolates the kernel's compute+load steady-state ceiling.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gemm_params.h"

typedef int8_t i8_t;
typedef int32_t i32_t;
typedef uint16_t bf16_t;
typedef float f32_t;

// exported pack + kernels
void i8_pack_B(const i8_t *B, i8_t *B_reo, int K, int N);
void i8_pack_A_neon_m8_asm(const i8_t *A, i8_t *P, int K_r, int lda);
void i8gemm_k_nld(const i8_t *A, const i8_t *B_reo, i32_t *C,
                  i8_t *A_reorder, const gemm_params_t *params);
void i8gemm_k_nld_m12(const i8_t *A, const i8_t *B_reo, i32_t *C,
                      i8_t *A_reorder, const gemm_params_t *params);
static void i8_pack_A_m12(const i8_t *A, i8_t *A_reo, int K_r) {
    size_t idx = 0;
    for (int kb = 0; kb < K_r; kb += 8)
        for (int rp = 0; rp < 6; rp++) {
            int r0 = rp * 2, r1 = r0 + 1;
            memcpy(A_reo + idx, A + (size_t)r0 * K_r + kb, 8); idx += 8;
            memcpy(A_reo + idx, A + (size_t)r1 * K_r + kb, 8); idx += 8;
        }
}

void bf16_pack_B(const bf16_t *B, bf16_t *B_reo, int K, int N);
void bf16gemm_k_nld_f_m12(const bf16_t *A, const bf16_t *B_reo, f32_t *C,
                          bf16_t *A_reorder, const gemm_params_t *params);

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static void *aa(size_t n) { void *p = aligned_alloc(64, (n + 63) & ~63ull); memset(p, 1, (n + 63) & ~63ull); return p; }

// replicate bf16 m12 A-pack (rowpairs=6, k step 4)
static void bf16_pack_A_m12(const bf16_t *A, bf16_t *A_reo, int M, int K_r) {
    int rowpairs = 6;
    size_t idx = 0;
    for (int kb = 0; kb < K_r; kb += 4)
        for (int rp = 0; rp < rowpairs; rp++) {
            int r0 = rp * 2, r1 = r0 + 1;
            for (int k = 0; k < 4; k++) A_reo[idx++] = (r0 < M) ? A[(size_t)r0 * K_r + kb + k] : 0;
            for (int k = 0; k < 4; k++) A_reo[idx++] = (r1 < M) ? A[(size_t)r1 * K_r + kb + k] : 0;
        }
}

int main(void) {
    const double i8_peak = 661.22, bf16_peak = 330.96;

    // ---- i8 main 8x2VL kernel, N tile = 16, K cache-resident ----
    {
        int M = 8, K = 1024, N = 16;
        i8_t *A = aa((size_t)M * K), *B = aa((size_t)K * N);
        i8_t *B_reo = aa((size_t)K * N);
        i8_t *A_reo = aa((size_t)K * 8);
        i32_t *C = aa((size_t)M * N * sizeof(i32_t) * 64);
        for (int i = 0; i < M * K; i++) A[i] = (i % 17) - 8;
        for (int i = 0; i < K * N; i++) B[i] = (i % 13) - 6;
        i8_pack_B(B, B_reo, K, N);
        i8_pack_A_neon_m8_asm(A, A_reo, K, K);
        gemm_params_t p = {M, K, N, K, K, N};
        long reps = 400000;
        for (int w = 0; w < 1000; w++) i8gemm_k_nld(A, B_reo, C, A_reo, &p);
        double best = 1e300;
        for (int r = 0; r < 5; r++) {
            double t0 = now_sec();
            for (long i = 0; i < reps; i++) i8gemm_k_nld(A, B_reo, C, A_reo, &p);
            double dt = now_sec() - t0; if (dt < best) best = dt;
        }
        double ops = 2.0 * M * K * N * (double)reps;
        double g = ops / best / 1e9;
        printf("prepacked i8  k_nld  (M8 N16 K%d, in-L1): %8.2f GOPS = %.1f%% of peak\n",
               K, g, g * 100.0 / i8_peak);
    }

    // ---- i8 m12 kernel (the big-shape path), N tile = 16, K in L1 ----
    {
        int M = 12, K = 1024, N = 16;
        i8_t *B = aa((size_t)K * N);
        i8_t *B_reo = aa((size_t)K * N);
        i8_t *Aflat = aa((size_t)M * K);
        i8_t *A_reo = aa((size_t)K * 12);
        i32_t *C = aa((size_t)M * N * sizeof(i32_t) * 64);
        for (int i = 0; i < M * K; i++) Aflat[i] = (i % 17) - 8;
        for (int i = 0; i < K * N; i++) B[i] = (i % 13) - 6;
        i8_pack_B(B, B_reo, K, N);
        i8_pack_A_m12(Aflat, A_reo, K);
        gemm_params_t p = {M, K, N, K, K, N};
        long reps = 400000;
        for (int w = 0; w < 1000; w++) i8gemm_k_nld_m12(Aflat, B_reo, C, A_reo, &p);
        double best = 1e300;
        for (int r = 0; r < 5; r++) {
            double t0 = now_sec();
            for (long i = 0; i < reps; i++) i8gemm_k_nld_m12(Aflat, B_reo, C, A_reo, &p);
            double dt = now_sec() - t0; if (dt < best) best = dt;
        }
        double ops = 2.0 * M * K * N * (double)reps;
        double g = ops / best / 1e9;
        printf("prepacked i8  m12    (M12 N16 K%d, in-L1): %8.2f GOPS = %.1f%% of peak\n",
               K, g, g * 100.0 / i8_peak);
    }

    // ---- bf16 m12 kernel, N tile = 16, K cache-resident ----
    {
        int M = 12, K = 1024, N = 16;
        bf16_t *A = aa((size_t)M * K * 2), *B = aa((size_t)K * N * 2);
        bf16_t *B_reo = aa((size_t)K * N * 2);
        bf16_t *A_reo = aa((size_t)K * 48);
        f32_t *C = aa((size_t)M * N * sizeof(f32_t) * 64);
        for (int i = 0; i < M * K; i++) A[i] = 0x3f80 + (i % 7);
        for (int i = 0; i < K * N; i++) B[i] = 0x3f80 + (i % 5);
        bf16_pack_B(B, B_reo, K, N);
        bf16_pack_A_m12(A, A_reo, M, K);
        gemm_params_t p = {M, K, N, K, K, N};
        long reps = 400000;
        for (int w = 0; w < 1000; w++) bf16gemm_k_nld_f_m12(A, B_reo, C, A_reo, &p);
        double best = 1e300;
        for (int r = 0; r < 5; r++) {
            double t0 = now_sec();
            for (long i = 0; i < reps; i++) bf16gemm_k_nld_f_m12(A, B_reo, C, A_reo, &p);
            double dt = now_sec() - t0; if (dt < best) best = dt;
        }
        double ops = 2.0 * M * K * N * (double)reps;
        double g = ops / best / 1e9;
        printf("prepacked bf16 m12   (M12 N16 K%d, in-L1): %8.2f GFLOPS = %.1f%% of peak\n",
               K, g, g * 100.0 / bf16_peak);
    }
    return 0;
}
