// bf16gemm_sve.c -- SVE BF16 GEMM implementation with the same public API.
//
// Link this file instead of bf16gemm_mt.c/bf16gemm_k*.S when building the
// SVE variant. The exported function names intentionally match the NEON path.

#include <arm_sve.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "gemm_params.h"

typedef uint16_t bf16_t;
typedef float f32_t;

static size_t round_up_64(size_t size) {
    return (size + 63u) & ~(size_t)63u;
}

static void *aligned_alloc_64(size_t size) {
    if (size == 0)
        size = 64;
    return aligned_alloc(64, round_up_64(size));
}

static int bf16_sve_segments(void) {
    return (int)(svcntb() / 16);
}

static int bf16_sve_n_tile(void) {
    return bf16_sve_segments() * 8;
}

static int round_up_int(int x, int q) {
    return ((x + q - 1) / q) * q;
}

static size_t bf16_a_reorder_stride(int K_r) {
    return (size_t)K_r * 8;
}

static void bf16_pack_A_sve_block(const bf16_t *A, bf16_t *A_reo,
                                  int M, int K_r) {
    int rowpairs = (M + 1) / 2;
    if (rowpairs < 4)
        rowpairs = 4;
    if (rowpairs > 6)
        rowpairs = 6;
    size_t idx = 0;

    for (int kb = 0; kb < K_r; kb += 4) {
        for (int rp = 0; rp < rowpairs; rp++) {
            int r0 = rp * 2;
            int r1 = r0 + 1;
            for (int k = 0; k < 4; k++)
                A_reo[idx++] = (r0 < M) ? A[(size_t)r0 * K_r + kb + k] : 0;
            for (int k = 0; k < 4; k++)
                A_reo[idx++] = (r1 < M) ? A[(size_t)r1 * K_r + kb + k] : 0;
        }
    }
}

static bf16_t *bf16_prepare_A_reorder_pool(const bf16_t *A, int M, int K_r,
                                           int num_threads) {
    const int blocks = (M + 7) / 8;
    const size_t stride = bf16_a_reorder_stride(K_r);
    bf16_t *pool = (bf16_t *)aligned_alloc_64((size_t)blocks * stride * sizeof(bf16_t));
    if (!pool)
        return NULL;

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int b = 0; b < blocks; b++) {
        int m0 = b * 8;
        int mb = M - m0 < 8 ? M - m0 : 8;
        bf16_pack_A_sve_block(A + (size_t)m0 * K_r, pool + (size_t)b * stride, mb, K_r);
    }
    return pool;
}

static bf16_t *bf16_prepare_A_reorder_pool_m12(const bf16_t *A, int blocks,
                                               int K_r, int num_threads) {
    const size_t stride = (size_t)K_r * 12;
    bf16_t *pool = (bf16_t *)aligned_alloc_64((size_t)blocks * stride * sizeof(bf16_t));
    if (!pool)
        return NULL;

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int b = 0; b < blocks; b++)
        bf16_pack_A_sve_block(A + (size_t)b * 12 * K_r,
                              pool + (size_t)b * stride, 12, K_r);
    return pool;
}

void bf16_pack_B(const bf16_t *B, bf16_t *B_reo, int K, int N) {
    const int segs = bf16_sve_segments();
    const int n_tile = segs * 8;
    int idx = 0;

    for (int nb = 0; nb < N; nb += n_tile) {
        for (int rb = 0; rb < K / 4; rb++) {
            int row_base = rb * 4;
            for (int cp = 0; cp < 4; cp++) {
                for (int sg = 0; sg < segs; sg++) {
                    int col_base = nb + sg * 8 + cp * 2;
                    for (int i = 0; i < 4; i++)
                        B_reo[idx++] = B[(row_base + i) * N + col_base];
                    for (int i = 0; i < 4; i++)
                        B_reo[idx++] = B[(row_base + i) * N + col_base + 1];
                }
            }
        }
    }
}

void bf16gemm_k_ld(const bf16_t *A, const bf16_t *B_reo, f32_t *C,
                   bf16_t *A_reorder, const gemm_params_t *params);
void bf16gemm_k_ld1(const bf16_t *A, const bf16_t *B_reo, f32_t *C,
                    bf16_t *A_reorder, const gemm_params_t *params);
void bf16gemm_k_ld2(const bf16_t *A, const bf16_t *B_reo, f32_t *C,
                    bf16_t *A_reorder, const gemm_params_t *params);
void bf16gemm_k_ld4(const bf16_t *A, const bf16_t *B_reo, f32_t *C,
                    bf16_t *A_reorder, const gemm_params_t *params);
void bf16gemm_k_ld_b(const bf16_t *A, const bf16_t *B_reo, bf16_t *C,
                     bf16_t *A_reorder, const gemm_params_t *params);
void bf16gemm_k_ld1_b(const bf16_t *A, const bf16_t *B_reo, bf16_t *C,
                      bf16_t *A_reorder, const gemm_params_t *params);
void bf16gemm_k_ld2_b(const bf16_t *A, const bf16_t *B_reo, bf16_t *C,
                      bf16_t *A_reorder, const gemm_params_t *params);
void bf16gemm_k_ld4_b(const bf16_t *A, const bf16_t *B_reo, bf16_t *C,
                      bf16_t *A_reorder, const gemm_params_t *params);
void bf16gemm_k_nld_b(const bf16_t *A, const bf16_t *B_reo, bf16_t *C,
                      bf16_t *A_reorder, const gemm_params_t *params);
void bf16gemm_k_nld1_b(const bf16_t *A, const bf16_t *B_reo, bf16_t *C,
                       bf16_t *A_reorder, const gemm_params_t *params);
void bf16gemm_k_nld2_b(const bf16_t *A, const bf16_t *B_reo, bf16_t *C,
                       bf16_t *A_reorder, const gemm_params_t *params);
void bf16gemm_k_nld4_b(const bf16_t *A, const bf16_t *B_reo, bf16_t *C,
                       bf16_t *A_reorder, const gemm_params_t *params);
void bf16gemm_k_ld_bias_f(const bf16_t *A, const bf16_t *B_reo, f32_t *C,
                          bf16_t *A_reorder, const gemm_params_t *params,
                          const f32_t *bias);
void bf16gemm_k_ld1_bias_f(const bf16_t *A, const bf16_t *B_reo, f32_t *C,
                           bf16_t *A_reorder, const gemm_params_t *params,
                           const f32_t *bias);
void bf16gemm_k_ld2_bias_f(const bf16_t *A, const bf16_t *B_reo, f32_t *C,
                           bf16_t *A_reorder, const gemm_params_t *params,
                           const f32_t *bias);
void bf16gemm_k_ld4_bias_f(const bf16_t *A, const bf16_t *B_reo, f32_t *C,
                           bf16_t *A_reorder, const gemm_params_t *params,
                           const f32_t *bias);
void bf16gemm_k_nld_f_m12(const bf16_t *A, const bf16_t *B_reo, f32_t *C,
                          bf16_t *A_reorder, const gemm_params_t *params);

static void bf16_dispatch_f32(const bf16_t *A, const bf16_t *B_reo,
                              f32_t *C, int M, int K_r, int N_r,
                              int ldc, int no_load, const f32_t *bias,
                              bf16_t *A_reorder) {
    gemm_params_t p = {M, K_r, N_r, K_r, K_r, ldc};
    if (bias) {
        if (M <= 1)
            bf16gemm_k_ld1_bias_f(A, B_reo, C, A_reorder, &p, bias);
        else if (M <= 2)
            bf16gemm_k_ld2_bias_f(A, B_reo, C, A_reorder, &p, bias);
        else if (M <= 4)
            bf16gemm_k_ld4_bias_f(A, B_reo, C, A_reorder, &p, bias);
        else
            bf16gemm_k_ld_bias_f(A, B_reo, C, A_reorder, &p, bias);
    } else {
        (void)no_load;
        if (M <= 1)
            bf16gemm_k_ld1(A, B_reo, C, A_reorder, &p);
        else if (M <= 2)
            bf16gemm_k_ld2(A, B_reo, C, A_reorder, &p);
        else if (M <= 4)
            bf16gemm_k_ld4(A, B_reo, C, A_reorder, &p);
        else
            bf16gemm_k_ld(A, B_reo, C, A_reorder, &p);
    }
}

static void bf16_dispatch_bf16(const bf16_t *A, const bf16_t *B_reo,
                               bf16_t *C, int M, int K_r, int N_r,
                               int ldc, int no_load, bf16_t *A_reorder) {
    gemm_params_t p = {M, K_r, N_r, K_r, K_r, ldc};
    if (no_load) {
        if (M <= 1)
            bf16gemm_k_nld1_b(A, B_reo, C, A_reorder, &p);
        else if (M <= 2)
            bf16gemm_k_nld2_b(A, B_reo, C, A_reorder, &p);
        else if (M <= 4)
            bf16gemm_k_nld4_b(A, B_reo, C, A_reorder, &p);
        else
            bf16gemm_k_nld_b(A, B_reo, C, A_reorder, &p);
    } else {
        if (M <= 1)
            bf16gemm_k_ld1_b(A, B_reo, C, A_reorder, &p);
        else if (M <= 2)
            bf16gemm_k_ld2_b(A, B_reo, C, A_reorder, &p);
        else if (M <= 4)
            bf16gemm_k_ld4_b(A, B_reo, C, A_reorder, &p);
        else
            bf16gemm_k_ld_b(A, B_reo, C, A_reorder, &p);
    }
}

void bf16gemm_mt_dispatch(const bf16_t *A, const bf16_t *B_reo,
                          f32_t *C, int M, int K_r, int N_r,
                          int num_threads) {
    if (num_threads <= 0)
        num_threads = omp_get_max_threads();
    if (num_threads <= 0)
        num_threads = 1;
    const int n_tile = bf16_sve_n_tile();
    const int n_tiles = N_r / n_tile;
    const size_t a_stride = bf16_a_reorder_stride(K_r);

    if (M / 8 >= num_threads || n_tiles < num_threads) {
        int blocks12 = M / 12;
        int m12_end = 0;
        bf16_t *A_reo12_pool = bf16_prepare_A_reorder_pool_m12(A, blocks12, K_r, num_threads);
        if (A_reo12_pool) {
            const size_t a12_stride = (size_t)K_r * 12;
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            for (int b = 0; b < blocks12; b++) {
                gemm_params_t p = {12, K_r, N_r, K_r, K_r, N_r};
                int m0 = b * 12;
                bf16gemm_k_nld_f_m12(A + (size_t)m0 * K_r, B_reo,
                                     C + (size_t)m0 * N_r,
                                     A_reo12_pool + (size_t)b * a12_stride, &p);
            }
            m12_end = blocks12 * 12;
        }

        int tail_M = M - m12_end;
        bf16_t *A_reo_pool = bf16_prepare_A_reorder_pool(A + (size_t)m12_end * K_r,
                                                         tail_M, K_r, num_threads);
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int m0 = m12_end; m0 < M; m0 += 8) {
            int mb = M - m0 < 8 ? M - m0 : 8;
            bf16_t *A_reo = A_reo_pool ? A_reo_pool + (size_t)((m0 - m12_end) / 8) * a_stride : NULL;
            bf16_dispatch_f32(A + (size_t)m0 * K_r, B_reo,
                              C + (size_t)m0 * N_r, mb, K_r, N_r, N_r, 1, NULL, A_reo);
        }
        free(A_reo_pool);
        free(A_reo12_pool);
    } else {
        bf16_t *A_reo_pool = bf16_prepare_A_reorder_pool(A, M, K_r, num_threads);
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int t = 0; t < n_tiles; t++) {
            int n0 = t * n_tile;
            for (int m0 = 0; m0 < M; m0 += 8) {
                int mb = M - m0 < 8 ? M - m0 : 8;
                bf16_t *A_reo = A_reo_pool ? A_reo_pool + (size_t)(m0 / 8) * a_stride : NULL;
                bf16_dispatch_f32(A + (size_t)m0 * K_r,
                                  B_reo + (size_t)t * K_r * n_tile,
                                  C + (size_t)m0 * N_r + n0,
                                  mb, K_r, n_tile, N_r, 1, NULL, A_reo);
            }
        }
        free(A_reo_pool);
    }
}

void bf16gemm_mt_dispatch_bias_f(const bf16_t *A, const bf16_t *B_reo,
                                 f32_t *C, int M, int K_r, int N_r,
                                 int num_threads, const f32_t *bias) {
    if (num_threads <= 0)
        num_threads = omp_get_max_threads();
    if (num_threads <= 0)
        num_threads = 1;
    const int n_tile = bf16_sve_n_tile();
    const int n_tiles = N_r / n_tile;
    const size_t a_stride = bf16_a_reorder_stride(K_r);
    bf16_t *A_reo_pool = bf16_prepare_A_reorder_pool(A, M, K_r, num_threads);

    if (M / 8 >= num_threads || n_tiles < num_threads) {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int m0 = 0; m0 < M; m0 += 8) {
            int mb = M - m0 < 8 ? M - m0 : 8;
            bf16_t *A_reo = A_reo_pool ? A_reo_pool + (size_t)(m0 / 8) * a_stride : NULL;
            bf16_dispatch_f32(A + (size_t)m0 * K_r, B_reo,
                              C + (size_t)m0 * N_r, mb, K_r, N_r, N_r, 1, bias, A_reo);
        }
    } else {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int t = 0; t < n_tiles; t++) {
            int n0 = t * n_tile;
            for (int m0 = 0; m0 < M; m0 += 8) {
                int mb = M - m0 < 8 ? M - m0 : 8;
                bf16_t *A_reo = A_reo_pool ? A_reo_pool + (size_t)(m0 / 8) * a_stride : NULL;
                bf16_dispatch_f32(A + (size_t)m0 * K_r,
                                  B_reo + (size_t)t * K_r * n_tile,
                                  C + (size_t)m0 * N_r + n0,
                                  mb, K_r, n_tile, N_r, 1, bias + n0, A_reo);
            }
        }
    }
    free(A_reo_pool);
}

void bf16gemm_mt_dispatch_nld_b(const bf16_t *A, const bf16_t *B_reo,
                                bf16_t *C, int M, int K_r, int N_r,
                                int num_threads) {
    if (num_threads <= 0)
        num_threads = omp_get_max_threads();
    if (num_threads <= 0)
        num_threads = 1;
    const int n_tile = bf16_sve_n_tile();
    const int n_tiles = N_r / n_tile;
    const size_t a_stride = bf16_a_reorder_stride(K_r);
    bf16_t *A_reo_pool = bf16_prepare_A_reorder_pool(A, M, K_r, num_threads);

    if (M / 8 >= num_threads || n_tiles < num_threads) {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int m0 = 0; m0 < M; m0 += 8) {
            int mb = M - m0 < 8 ? M - m0 : 8;
            bf16_t *A_reo = A_reo_pool ? A_reo_pool + (size_t)(m0 / 8) * a_stride : NULL;
            bf16_dispatch_bf16(A + (size_t)m0 * K_r, B_reo,
                               C + (size_t)m0 * N_r, mb, K_r, N_r, N_r, 1, A_reo);
        }
    } else {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int t = 0; t < n_tiles; t++) {
            int n0 = t * n_tile;
            for (int m0 = 0; m0 < M; m0 += 8) {
                int mb = M - m0 < 8 ? M - m0 : 8;
                bf16_t *A_reo = A_reo_pool ? A_reo_pool + (size_t)(m0 / 8) * a_stride : NULL;
                bf16_dispatch_bf16(A + (size_t)m0 * K_r,
                                   B_reo + (size_t)t * K_r * n_tile,
                                   C + (size_t)m0 * N_r + n0,
                                   mb, K_r, n_tile, N_r, 1, A_reo);
            }
        }
    }
    free(A_reo_pool);
}

static int bf16_prepare(const bf16_t *A_orig, const bf16_t *B_orig,
                        bf16_t **A_pad, bf16_t **B_reo,
                        int M, int K, int N, int *K_r, int *N_r) {
    *K_r = round_up_int(K < 8 ? 8 : K, 8);
    *N_r = round_up_int(N < 8 ? 8 : N, bf16_sve_n_tile());
    *A_pad = NULL;
    *B_reo = NULL;

    bf16_t *B_pad = (bf16_t *)calloc((size_t)*K_r * *N_r, sizeof(bf16_t));
    if (!B_pad)
        return 0;
    for (int i = 0; i < K; i++)
        memcpy(B_pad + (size_t)i * *N_r, B_orig + (size_t)i * N, (size_t)N * sizeof(bf16_t));

    *B_reo = (bf16_t *)aligned_alloc_64((size_t)*K_r * *N_r * sizeof(bf16_t));
    if (!*B_reo) {
        free(B_pad);
        return 0;
    }
    bf16_pack_B(B_pad, *B_reo, *K_r, *N_r);
    free(B_pad);

    if (*K_r == K) {
        *A_pad = (bf16_t *)A_orig;
    } else {
        *A_pad = (bf16_t *)calloc((size_t)M * *K_r, sizeof(bf16_t));
        if (!*A_pad) {
            free(*B_reo);
            return 0;
        }
        for (int i = 0; i < M; i++)
            memcpy(*A_pad + (size_t)i * *K_r, A_orig + (size_t)i * K, (size_t)K * sizeof(bf16_t));
    }
    return 1;
}

void bf16gemm_mt(const bf16_t *A_orig, const bf16_t *B_orig,
                 f32_t *C, int M, int K, int N, int num_threads) {
    bf16_t *A_use, *B_reo;
    int K_r, N_r;
    if (!bf16_prepare(A_orig, B_orig, &A_use, &B_reo, M, K, N, &K_r, &N_r))
        return;
    f32_t *C_pad = (f32_t *)calloc((size_t)M * N_r, sizeof(f32_t));
    if (C_pad) {
        bf16gemm_mt_dispatch(A_use, B_reo, C_pad, M, K_r, N_r, num_threads);
        for (int i = 0; i < M; i++)
            memcpy(C + (size_t)i * N, C_pad + (size_t)i * N_r, (size_t)N * sizeof(f32_t));
        free(C_pad);
    }
    if (A_use != A_orig)
        free(A_use);
    free(B_reo);
}

void bf16gemm_mt_bias_f(const bf16_t *A_orig, const bf16_t *B_orig,
                        f32_t *C, int M, int K, int N, int num_threads,
                        const f32_t *bias) {
    bf16_t *A_use, *B_reo;
    int K_r, N_r;
    if (!bf16_prepare(A_orig, B_orig, &A_use, &B_reo, M, K, N, &K_r, &N_r))
        return;
    f32_t *bias_pad = (f32_t *)calloc((size_t)N_r, sizeof(f32_t));
    f32_t *C_pad = (f32_t *)calloc((size_t)M * N_r, sizeof(f32_t));
    if (bias_pad && C_pad) {
        memcpy(bias_pad, bias, (size_t)N * sizeof(f32_t));
        bf16gemm_mt_dispatch_bias_f(A_use, B_reo, C_pad, M, K_r, N_r,
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

void bf16gemm_mt_nld_b(const bf16_t *A_orig, const bf16_t *B_orig,
                       bf16_t *C, int M, int K, int N, int num_threads) {
    bf16_t *A_use, *B_reo;
    int K_r, N_r;
    if (!bf16_prepare(A_orig, B_orig, &A_use, &B_reo, M, K, N, &K_r, &N_r))
        return;
    bf16_t *C_pad = (bf16_t *)calloc((size_t)M * N_r, sizeof(bf16_t));
    if (C_pad) {
        bf16gemm_mt_dispatch_nld_b(A_use, B_reo, C_pad, M, K_r, N_r, num_threads);
        for (int i = 0; i < M; i++)
            memcpy(C + (size_t)i * N, C_pad + (size_t)i * N_r, (size_t)N * sizeof(bf16_t));
        free(C_pad);
    }
    if (A_use != A_orig)
        free(A_use);
    free(B_reo);
}
