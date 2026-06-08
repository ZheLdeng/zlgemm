// test_sve_m8_interfaces.c -- focused correctness test for M8 BF16 entry points.
//
// Build:
//   cc -O2 -Wall -march=armv8.6-a+sve+bf16+i8mm test_sve_m8_interfaces.c bf16gemm_sve_m8_nld.S -lm -o test_sve_m8_interfaces
//   cc -O2 -Wall -march=armv8.6-a+sve+bf16+i8mm -DM8_NEON_PACK=1 test_sve_m8_interfaces.c bf16gemm_neon_m8_nld.S -lm -o test_neon_m8_interfaces

#include <arm_sve.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gemm_params.h"

typedef uint16_t bf16_t;

void bf16gemm_k_nld_f_m8(const bf16_t *A, const bf16_t *B_reo,
                         float *C, bf16_t *A_reorder,
                         const gemm_params_t *params);
void bf16gemm_k_nld_b_m8(const bf16_t *A, const bf16_t *B_reo,
                         bf16_t *C, bf16_t *A_reorder,
                         const gemm_params_t *params);
void bf16gemm_k_nld_bias_f_m8(const bf16_t *A, const bf16_t *B_reo,
                              float *C, bf16_t *A_reorder,
                              const gemm_params_t *params,
                              const float *bias);

static bf16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    u += ((u >> 16) & 1u) + 0x7fffu;
    return (bf16_t)(u >> 16);
}

static float bf16_to_f32(bf16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static void *xcalloc(size_t n, size_t size) {
    void *p = calloc(n, size);
    if (!p) {
        perror("calloc");
        exit(1);
    }
    return p;
}

static void fill_bf16(bf16_t *p, size_t n, int salt) {
    for (size_t i = 0; i < n; i++) {
        int v = (int)((i * 37u + (unsigned)salt * 17u) % 31u) - 15;
        p[i] = f32_to_bf16((float)v * 0.0625f);
    }
}

static void fill_bias(float *p, size_t n) {
    for (size_t i = 0; i < n; i++)
        p[i] = (float)((int)(i % 13u) - 6) * 0.125f;
}

#ifndef M8_NEON_PACK
static void pack_b_sve(const bf16_t *B, bf16_t *P, int K, int N) {
    const int segs = (int)svcntb() / 16;
    const int n_tile = segs * 8;
    size_t idx = 0;

    for (int nb = 0; nb < N; nb += n_tile) {
        for (int rb = 0; rb < K / 4; rb++) {
            int row_base = rb * 4;
            for (int cp = 0; cp < 4; cp++) {
                for (int sg = 0; sg < segs; sg++) {
                    int col_base = nb + sg * 8 + cp * 2;
                    for (int i = 0; i < 4; i++)
                        P[idx++] = B[(size_t)(row_base + i) * N + col_base];
                    for (int i = 0; i < 4; i++)
                        P[idx++] = B[(size_t)(row_base + i) * N + col_base + 1];
                }
            }
        }
    }
}
#endif

#ifdef M8_NEON_PACK
static void pack_b_neon(const bf16_t *B, bf16_t *P, int K, int N) {
    size_t idx = 0;
    for (int cb = 0; cb < N / 8; cb++) {
        for (int rb = 0; rb < K / 4; rb++) {
            int row_base = rb * 4;
            int col_base = cb * 8;
            for (int cp = 0; cp < 4; cp++) {
                int c0 = col_base + cp * 2;
                int c1 = c0 + 1;
                for (int i = 0; i < 4; i++)
                    P[idx++] = B[(size_t)(row_base + i) * N + c0];
                for (int i = 0; i < 4; i++)
                    P[idx++] = B[(size_t)(row_base + i) * N + c1];
            }
        }
    }
}
#endif

static void ref_gemm(const bf16_t *A, const bf16_t *B, float *C,
                     int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += bf16_to_f32(A[(size_t)i * K + k]) *
                       bf16_to_f32(B[(size_t)k * N + j]);
            C[(size_t)i * N + j] = sum;
        }
    }
}

static int check_f32(const float *got, const float *ref, const float *bias,
                     int M, int N, int ldc, const char *name) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float expect = ref[(size_t)i * N + j] + (bias ? bias[j] : 0.0f);
            float err = fabsf(got[(size_t)i * ldc + j] - expect);
            if (err > 0.5f) {
                fprintf(stderr,
                        "%s mismatch (%d,%d): got=%f expect=%f err=%f\n",
                        name, i, j, got[(size_t)i * ldc + j], expect, err);
                return 0;
            }
        }
    }
    return 1;
}

static int check_bf16(const bf16_t *got, const float *ref,
                      int M, int N, int ldc, const char *name) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float g = bf16_to_f32(got[(size_t)i * ldc + j]);
            float err = fabsf(g - ref[(size_t)i * N + j]);
            if (err > 0.5f) {
                fprintf(stderr,
                        "%s mismatch (%d,%d): got=%f expect=%f err=%f\n",
                        name, i, j, g, ref[(size_t)i * N + j], err);
                return 0;
            }
        }
    }
    return 1;
}

static int run_case(int M, int K, int N, int ldc) {
    bf16_t *A = (bf16_t *)xcalloc((size_t)M * K, sizeof(*A));
    bf16_t *B = (bf16_t *)xcalloc((size_t)K * N, sizeof(*B));
    bf16_t *B_reo = (bf16_t *)xcalloc((size_t)K * N, sizeof(*B_reo));
    bf16_t *A_reorder = (bf16_t *)xcalloc((size_t)M * K, sizeof(*A_reorder));
    float *ref = (float *)xcalloc((size_t)M * N, sizeof(*ref));
    float *bias = (float *)xcalloc((size_t)N, sizeof(*bias));
    float *C_f32 = (float *)xcalloc((size_t)M * ldc, sizeof(*C_f32));
    float *C_bias = (float *)xcalloc((size_t)M * ldc, sizeof(*C_bias));
    bf16_t *C_bf16 = (bf16_t *)xcalloc((size_t)M * ldc, sizeof(*C_bf16));
    gemm_params_t p = {M, K, N, K, K, ldc};
    int ok = 1;

    fill_bf16(A, (size_t)M * K, 1);
    fill_bf16(B, (size_t)K * N, 2);
    fill_bias(bias, (size_t)N);
#ifdef M8_NEON_PACK
    pack_b_neon(B, B_reo, K, N);
#else
    pack_b_sve(B, B_reo, K, N);
#endif
    ref_gemm(A, B, ref, M, K, N);

    bf16gemm_k_nld_f_m8(A, B_reo, C_f32, A_reorder, &p);
    memset(A_reorder, 0, (size_t)M * K * sizeof(*A_reorder));
    bf16gemm_k_nld_b_m8(A, B_reo, C_bf16, A_reorder, &p);
    memset(A_reorder, 0, (size_t)M * K * sizeof(*A_reorder));
    bf16gemm_k_nld_bias_f_m8(A, B_reo, C_bias, A_reorder, &p, bias);

    ok = ok && check_f32(C_f32, ref, NULL, M, N, ldc, "f32");
    ok = ok && check_bf16(C_bf16, ref, M, N, ldc, "bf16");
    ok = ok && check_f32(C_bias, ref, bias, M, N, ldc, "bias_f32");

    free(A);
    free(B);
    free(B_reo);
    free(A_reorder);
    free(ref);
    free(bias);
    free(C_f32);
    free(C_bias);
    free(C_bf16);
    return ok;
}

int main(void) {
#ifdef M8_NEON_PACK
    const int n_tile = 8;
#else
    const int n_tile = (int)svcntb() / 2;
#endif
    int ok = 1;

    ok = ok && run_case(8, 8, n_tile, n_tile + 8);
    ok = ok && run_case(16, 16, n_tile * 2, n_tile * 2 + 8);

    if (!ok)
        return 1;
    puts("m8_interfaces: ok");
    return 0;
}
