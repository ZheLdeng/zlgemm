// i8gemm_sve.c -- SVE I8 GEMM implementation with the same public API.
//
// Link this file instead of i8gemm_mt.c/i8gemm_k*.S when building the SVE
// variant. The exported function names intentionally match the NEON path.

#include <arm_sve.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "gemm_params.h"

typedef int8_t i8_t;
typedef int32_t i32_t;
typedef float f32_t;

static size_t round_up_64(size_t size) {
    return (size + 63u) & ~(size_t)63u;
}

static void *aligned_alloc_64(size_t size) {
    if (size == 0)
        size = 64;
    return aligned_alloc(64, round_up_64(size));
}

static int i8_sve_segments(void) {
    return (int)(svcntb() / 16);
}

static int i8_sve_n_tile(void) {
    return i8_sve_segments() * 8;
}

static int round_up_int(int x, int q) {
    return ((x + q - 1) / q) * q;
}

static int i8_sve_effective_threads(int num_threads, int M, int n_tiles) {
    if (num_threads <= 0)
        num_threads = omp_get_max_threads();
    if (num_threads <= 0)
        num_threads = 1;

    const char *clamp_env = getenv("I8_SVE_CLAMP_THREADS");
    int clamp_threads = clamp_env ? atoi(clamp_env) != 0 : 1;
    if (!clamp_threads)
        return num_threads;

    int work_units = ((M + 7) / 8) * n_tiles;
    if (work_units < 1)
        work_units = 1;
    if (num_threads > work_units)
        num_threads = work_units;

    if (M < 64 && n_tiles < num_threads * 4) {
        int by_n = (n_tiles + 3) / 4;
        if (by_n < 1)
            by_n = 1;
        if (num_threads > by_n)
            num_threads = by_n;
    }
    return num_threads;
}

static int i8_sve_use_n_split(int M, int K_r, int N_r, int n_tiles,
                              int num_threads) {
    const char *split = getenv("I8_SVE_SPLIT");
    if (split) {
        if (strcmp(split, "m") == 0)
            return 0;
        if (strcmp(split, "n") == 0)
            return n_tiles >= num_threads;
    }

    if (num_threads <= 1 || n_tiles < num_threads)
        return 0;
    if (M / 8 < num_threads)
        return 1;

    const size_t b_panel_bytes = (size_t)K_r * (size_t)N_r * sizeof(i8_t);
    return b_panel_bytes >= 512u * 1024u && N_r >= M * 2;
}

void i8_pack_B(const i8_t *B, i8_t *B_reo, int K, int N) {
    const int segs = i8_sve_segments();
    const int n_tile = segs * 8;
    int idx = 0;

    for (int nb = 0; nb < N; nb += n_tile) {
        for (int rb = 0; rb < K / 8; rb++) {
            int row_base = rb * 8;
            for (int cp = 0; cp < 4; cp++) {
                for (int sg = 0; sg < segs; sg++) {
                    int col_base = nb + sg * 8 + cp * 2;
                    for (int j = 0; j < 2; j++)
                        for (int i = 0; i < 8; i++)
                            B_reo[idx++] = B[(row_base + i) * N + col_base + j];
                }
            }
        }
    }
}


void i8gemm_k_ld(const i8_t *A, const i8_t *B_reo, i32_t *C,
                 i8_t *A_reorder, const gemm_params_t *params);
void i8gemm_k_ld1(const i8_t *A, const i8_t *B_reo, i32_t *C,
                  i8_t *A_reorder, const gemm_params_t *params);
void i8gemm_k_ld2(const i8_t *A, const i8_t *B_reo, i32_t *C,
                  i8_t *A_reorder, const gemm_params_t *params);
void i8gemm_k_ld4(const i8_t *A, const i8_t *B_reo, i32_t *C,
                  i8_t *A_reorder, const gemm_params_t *params);
void i8gemm_k_ld_f(const i8_t *A, const i8_t *B_reo, f32_t *C,
                   i8_t *A_reorder, const gemm_params_t *params);
void i8gemm_k_ld1_f(const i8_t *A, const i8_t *B_reo, f32_t *C,
                    i8_t *A_reorder, const gemm_params_t *params);
void i8gemm_k_ld2_f(const i8_t *A, const i8_t *B_reo, f32_t *C,
                    i8_t *A_reorder, const gemm_params_t *params);
void i8gemm_k_ld4_f(const i8_t *A, const i8_t *B_reo, f32_t *C,
                    i8_t *A_reorder, const gemm_params_t *params);
void i8gemm_k_ld_bias_f(const i8_t *A, const i8_t *B_reo, f32_t *C,
                        i8_t *A_reorder, const gemm_params_t *params,
                        const f32_t *bias);
void i8gemm_k_ld1_bias_f(const i8_t *A, const i8_t *B_reo, f32_t *C,
                         i8_t *A_reorder, const gemm_params_t *params,
                         const f32_t *bias);
void i8gemm_k_ld2_bias_f(const i8_t *A, const i8_t *B_reo, f32_t *C,
                         i8_t *A_reorder, const gemm_params_t *params,
                         const f32_t *bias);
void i8gemm_k_ld4_bias_f(const i8_t *A, const i8_t *B_reo, f32_t *C,
                         i8_t *A_reorder, const gemm_params_t *params,
                         const f32_t *bias);

static void i8_dispatch_i32(const i8_t *A, const i8_t *B_reo, i32_t *C,
                            int M, int K_r, int N_r, int ldc) {
    gemm_params_t p = {M, K_r, N_r, K_r, K_r, ldc};
    i8gemm_k_ld(A, B_reo, C, NULL, &p);
}

static void i8_dispatch_f32(const i8_t *A, const i8_t *B_reo, f32_t *C,
                            int M, int K_r, int N_r, int ldc,
                            const f32_t *bias) {
    gemm_params_t p = {M, K_r, N_r, K_r, K_r, ldc};
    if (bias)
        i8gemm_k_ld_bias_f(A, B_reo, C, NULL, &p, bias);
    else
        i8gemm_k_ld_f(A, B_reo, C, NULL, &p);
}

void i8gemm_mt_dispatch(const i8_t *A, const i8_t *B_reo,
                        i32_t *C, int M, int K_r, int N_r,
                        int num_threads) {
    const int n_tile = i8_sve_n_tile();
    const int n_tiles = N_r / n_tile;
    num_threads = i8_sve_effective_threads(num_threads, M, n_tiles);
    const int use_n_split =
        i8_sve_use_n_split(M, K_r, N_r, n_tiles, num_threads);

    if (!use_n_split) {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int m0 = 0; m0 < M; m0 += 8) {
            int mb = M - m0 < 8 ? M - m0 : 8;
            i8_dispatch_i32(A + (size_t)m0 * K_r, B_reo,
                            C + (size_t)m0 * N_r, mb, K_r, N_r, N_r);
        }
    } else {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int t = 0; t < n_tiles; t++) {
            int n0 = t * n_tile;
            i8_dispatch_i32(A, B_reo + (size_t)t * K_r * n_tile,
                            C + n0, M, K_r, n_tile, N_r);
        }
    }
}

void i8gemm_mt_dispatch_f(const i8_t *A, const i8_t *B_reo,
                          f32_t *C, int M, int K_r, int N_r,
                          int num_threads) {
    const int n_tile = i8_sve_n_tile();
    const int n_tiles = N_r / n_tile;
    num_threads = i8_sve_effective_threads(num_threads, M, n_tiles);
    const int use_n_split =
        i8_sve_use_n_split(M, K_r, N_r, n_tiles, num_threads);

    if (!use_n_split) {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int m0 = 0; m0 < M; m0 += 8) {
            int mb = M - m0 < 8 ? M - m0 : 8;
            i8_dispatch_f32(A + (size_t)m0 * K_r, B_reo,
                            C + (size_t)m0 * N_r, mb, K_r, N_r, N_r, NULL);
        }
    } else {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int t = 0; t < n_tiles; t++) {
            int n0 = t * n_tile;
            i8_dispatch_f32(A, B_reo + (size_t)t * K_r * n_tile,
                            C + n0, M, K_r, n_tile, N_r, NULL);
        }
    }
}

void i8gemm_mt_dispatch_bias_f(const i8_t *A, const i8_t *B_reo,
                               f32_t *C, int M, int K_r, int N_r,
                               int num_threads, const f32_t *bias) {
    const int n_tile = i8_sve_n_tile();
    const int n_tiles = N_r / n_tile;
    num_threads = i8_sve_effective_threads(num_threads, M, n_tiles);
    const int use_n_split =
        i8_sve_use_n_split(M, K_r, N_r, n_tiles, num_threads);

    if (!use_n_split) {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int m0 = 0; m0 < M; m0 += 8) {
            int mb = M - m0 < 8 ? M - m0 : 8;
            i8_dispatch_f32(A + (size_t)m0 * K_r, B_reo,
                            C + (size_t)m0 * N_r, mb, K_r, N_r, N_r, bias);
        }
    } else {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int t = 0; t < n_tiles; t++) {
            int n0 = t * n_tile;
            i8_dispatch_f32(A, B_reo + (size_t)t * K_r * n_tile,
                            C + n0, M, K_r, n_tile, N_r, bias + n0);
        }
    }
}

static int i8_prepare(const i8_t *A_orig, const i8_t *B_orig,
                      i8_t **A_pad, i8_t **B_reo,
                      int M, int K, int N, int *K_r, int *N_r) {
    *K_r = round_up_int(K < 16 ? 16 : K, 16);
    *N_r = round_up_int(N < 8 ? 8 : N, i8_sve_n_tile());
    *A_pad = NULL;
    *B_reo = NULL;

    i8_t *B_pad = (i8_t *)calloc((size_t)*K_r * *N_r, sizeof(i8_t));
    if (!B_pad)
        return 0;
    for (int i = 0; i < K; i++)
        memcpy(B_pad + (size_t)i * *N_r, B_orig + (size_t)i * N, (size_t)N);

    *B_reo = (i8_t *)aligned_alloc_64((size_t)*K_r * *N_r);
    if (!*B_reo) {
        free(B_pad);
        return 0;
    }
    i8_pack_B(B_pad, *B_reo, *K_r, *N_r);
    free(B_pad);

    if (*K_r == K) {
        *A_pad = (i8_t *)A_orig;
    } else {
        *A_pad = (i8_t *)calloc((size_t)M * *K_r, sizeof(i8_t));
        if (!*A_pad) {
            free(*B_reo);
            return 0;
        }
        for (int i = 0; i < M; i++)
            memcpy(*A_pad + (size_t)i * *K_r, A_orig + (size_t)i * K, (size_t)K);
    }
    return 1;
}

void i8gemm_mt(const i8_t *A_orig, const i8_t *B_orig,
               i32_t *C, int M, int K, int N, int num_threads) {
    i8_t *A_use, *B_reo;
    int K_r, N_r;
    if (!i8_prepare(A_orig, B_orig, &A_use, &B_reo, M, K, N, &K_r, &N_r))
        return;
    i32_t *C_pad = (i32_t *)calloc((size_t)M * N_r, sizeof(i32_t));
    if (C_pad) {
        i8gemm_mt_dispatch(A_use, B_reo, C_pad, M, K_r, N_r, num_threads);
        for (int i = 0; i < M; i++)
            memcpy(C + (size_t)i * N, C_pad + (size_t)i * N_r, (size_t)N * sizeof(i32_t));
        free(C_pad);
    }
    if (A_use != A_orig)
        free(A_use);
    free(B_reo);
}

void i8gemm_mt_f(const i8_t *A_orig, const i8_t *B_orig,
                 f32_t *C, int M, int K, int N, int num_threads) {
    i8_t *A_use, *B_reo;
    int K_r, N_r;
    if (!i8_prepare(A_orig, B_orig, &A_use, &B_reo, M, K, N, &K_r, &N_r))
        return;
    f32_t *C_pad = (f32_t *)calloc((size_t)M * N_r, sizeof(f32_t));
    if (C_pad) {
        i8gemm_mt_dispatch_f(A_use, B_reo, C_pad, M, K_r, N_r, num_threads);
        for (int i = 0; i < M; i++)
            memcpy(C + (size_t)i * N, C_pad + (size_t)i * N_r, (size_t)N * sizeof(f32_t));
        free(C_pad);
    }
    if (A_use != A_orig)
        free(A_use);
    free(B_reo);
}

void i8gemm_mt_bias_f(const i8_t *A_orig, const i8_t *B_orig,
                      f32_t *C, int M, int K, int N,
                      int num_threads, const f32_t *bias) {
    i8_t *A_use, *B_reo;
    int K_r, N_r;
    if (!i8_prepare(A_orig, B_orig, &A_use, &B_reo, M, K, N, &K_r, &N_r))
        return;
    f32_t *bias_pad = (f32_t *)calloc((size_t)N_r, sizeof(f32_t));
    f32_t *C_pad = (f32_t *)calloc((size_t)M * N_r, sizeof(f32_t));
    if (bias_pad && C_pad) {
        memcpy(bias_pad, bias, (size_t)N * sizeof(f32_t));
        i8gemm_mt_dispatch_bias_f(A_use, B_reo, C_pad, M, K_r, N_r,
                                  num_threads, bias_pad);
        for (int i = 0; i < M; i++)
            memcpy(C + (size_t)i * N, C_pad + (size_t)i * N_r, (size_t)N * sizeof(f32_t));
    }
    free(bias_pad);
    free(C_pad);
    if (A_use != A_orig)
        free(A_use);
    free(B_reo);
}
