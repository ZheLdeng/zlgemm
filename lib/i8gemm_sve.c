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

static void i8_sve_kernel(const i8_t *A, const i8_t *B_reo,
                          void *C, const gemm_params_t *params,
                          int max_m, int load_c, int store_f32,
                          const f32_t *bias) {
    const int M = params->m;
    const int K = params->k;
    const int N = params->n;
    const int lda = params->lda;
    const int ldc = params->ldc;
    const int segs = i8_sve_segments();
    const int n_tile = segs * 8;
    svbool_t pg_b8 = svptrue_b8();
    svbool_t pg_i32 = svptrue_b32();

    for (int m0 = 0; m0 < M; m0 += max_m) {
        int m_now = M - m0;
        if (m_now > max_m)
            m_now = max_m;

        for (int n0 = 0; n0 < N; n0 += n_tile) {
            svint32_t acc00, acc01, acc02, acc03;
            svint32_t acc10, acc11, acc12, acc13;
            svint32_t acc20, acc21, acc22, acc23;
            svint32_t acc30, acc31, acc32, acc33;

#define I8_INIT_ACC(VAR, RP, CP) do {                                   \
                int r0_ = (RP) * 2;                                     \
                int r1_ = r0_ + 1;                                      \
                if (load_c) {                                           \
                    i32_t tmp_[64];                                     \
                    memset(tmp_, 0, sizeof(tmp_));                       \
                    for (int sg = 0; sg < segs; sg++) {                 \
                        int c0_ = n0 + sg * 8 + (CP) * 2;               \
                        tmp_[sg * 4 + 0] = (r0_ < m_now) ? ((i32_t *)C)[(m0 + r0_) * (size_t)ldc + c0_] : 0; \
                        tmp_[sg * 4 + 1] = (r0_ < m_now) ? ((i32_t *)C)[(m0 + r0_) * (size_t)ldc + c0_ + 1] : 0; \
                        tmp_[sg * 4 + 2] = (r1_ < m_now) ? ((i32_t *)C)[(m0 + r1_) * (size_t)ldc + c0_] : 0; \
                        tmp_[sg * 4 + 3] = (r1_ < m_now) ? ((i32_t *)C)[(m0 + r1_) * (size_t)ldc + c0_ + 1] : 0; \
                    }                                                   \
                    VAR = svld1_s32(pg_i32, tmp_);                     \
                } else {                                                \
                    VAR = svdup_s32(0);                                 \
                }                                                       \
            } while (0)

            I8_INIT_ACC(acc00, 0, 0);
            I8_INIT_ACC(acc01, 0, 1);
            I8_INIT_ACC(acc02, 0, 2);
            I8_INIT_ACC(acc03, 0, 3);
            I8_INIT_ACC(acc10, 1, 0);
            I8_INIT_ACC(acc11, 1, 1);
            I8_INIT_ACC(acc12, 1, 2);
            I8_INIT_ACC(acc13, 1, 3);
            I8_INIT_ACC(acc20, 2, 0);
            I8_INIT_ACC(acc21, 2, 1);
            I8_INIT_ACC(acc22, 2, 2);
            I8_INIT_ACC(acc23, 2, 3);
            I8_INIT_ACC(acc30, 3, 0);
            I8_INIT_ACC(acc31, 3, 1);
            I8_INIT_ACC(acc32, 3, 2);
            I8_INIT_ACC(acc33, 3, 3);

            const i8_t *B_tile = B_reo + (size_t)(n0 / n_tile) * K * n_tile;
            for (int kb = 0; kb < K; kb += 8) {
                size_t boff0 = (((size_t)(kb / 8) * 4 + 0) * segs) * 16;
                size_t boff1 = (((size_t)(kb / 8) * 4 + 1) * segs) * 16;
                size_t boff2 = (((size_t)(kb / 8) * 4 + 2) * segs) * 16;
                size_t boff3 = (((size_t)(kb / 8) * 4 + 3) * segs) * 16;
                svint8_t b0 = svld1_s8(pg_b8, B_tile + boff0);
                svint8_t b1 = svld1_s8(pg_b8, B_tile + boff1);
                svint8_t b2 = svld1_s8(pg_b8, B_tile + boff2);
                svint8_t b3 = svld1_s8(pg_b8, B_tile + boff3);

#define I8_UPDATE_ROW(RP, A0, A1, A2, A3) do {                          \
                    int r0_ = (RP) * 2;                                 \
                    int r1_ = r0_ + 1;                                  \
                    i8_t a_tmp_[16] = {0};                              \
                    if (r0_ < m_now)                                    \
                        memcpy(a_tmp_, A + (size_t)(m0 + r0_) * lda + kb, 8); \
                    if (r1_ < m_now)                                    \
                        memcpy(a_tmp_ + 8, A + (size_t)(m0 + r1_) * lda + kb, 8); \
                    svint8_t az_ = svld1rq_s8(pg_b8, a_tmp_);           \
                    A0 = svmmla_s32(A0, az_, b0);                       \
                    A1 = svmmla_s32(A1, az_, b1);                       \
                    A2 = svmmla_s32(A2, az_, b2);                       \
                    A3 = svmmla_s32(A3, az_, b3);                       \
                } while (0)

                I8_UPDATE_ROW(0, acc00, acc01, acc02, acc03);
                I8_UPDATE_ROW(1, acc10, acc11, acc12, acc13);
                I8_UPDATE_ROW(2, acc20, acc21, acc22, acc23);
                I8_UPDATE_ROW(3, acc30, acc31, acc32, acc33);
            }

#define I8_STORE_ACC(VAR, RP, CP) do {                                  \
                int r0_ = (RP) * 2;                                     \
                int r1_ = r0_ + 1;                                      \
                i32_t tmp_[64];                                         \
                svst1_s32(pg_i32, tmp_, VAR);                           \
                for (int sg = 0; sg < segs; sg++) {                     \
                    int c0_ = n0 + sg * 8 + (CP) * 2;                   \
                    i32_t v00_ = tmp_[sg * 4 + 0];                      \
                    i32_t v01_ = tmp_[sg * 4 + 1];                      \
                    i32_t v10_ = tmp_[sg * 4 + 2];                      \
                    i32_t v11_ = tmp_[sg * 4 + 3];                      \
                    if (store_f32) {                                    \
                        f32_t *Cf_ = (f32_t *)C;                        \
                        if (r0_ < m_now) {                              \
                            Cf_[(m0 + r0_) * (size_t)ldc + c0_] = (f32_t)v00_ + (bias ? bias[c0_] : 0.0f); \
                            Cf_[(m0 + r0_) * (size_t)ldc + c0_ + 1] = (f32_t)v01_ + (bias ? bias[c0_ + 1] : 0.0f); \
                        }                                               \
                        if (r1_ < m_now) {                              \
                            Cf_[(m0 + r1_) * (size_t)ldc + c0_] = (f32_t)v10_ + (bias ? bias[c0_] : 0.0f); \
                            Cf_[(m0 + r1_) * (size_t)ldc + c0_ + 1] = (f32_t)v11_ + (bias ? bias[c0_ + 1] : 0.0f); \
                        }                                               \
                    } else {                                            \
                        i32_t *Ci_ = (i32_t *)C;                        \
                        if (r0_ < m_now) {                              \
                            Ci_[(m0 + r0_) * (size_t)ldc + c0_] = v00_; \
                            Ci_[(m0 + r0_) * (size_t)ldc + c0_ + 1] = v01_; \
                        }                                               \
                        if (r1_ < m_now) {                              \
                            Ci_[(m0 + r1_) * (size_t)ldc + c0_] = v10_; \
                            Ci_[(m0 + r1_) * (size_t)ldc + c0_ + 1] = v11_; \
                        }                                               \
                    }                                                   \
                }                                                       \
            } while (0)

            I8_STORE_ACC(acc00, 0, 0);
            I8_STORE_ACC(acc01, 0, 1);
            I8_STORE_ACC(acc02, 0, 2);
            I8_STORE_ACC(acc03, 0, 3);
            I8_STORE_ACC(acc10, 1, 0);
            I8_STORE_ACC(acc11, 1, 1);
            I8_STORE_ACC(acc12, 1, 2);
            I8_STORE_ACC(acc13, 1, 3);
            I8_STORE_ACC(acc20, 2, 0);
            I8_STORE_ACC(acc21, 2, 1);
            I8_STORE_ACC(acc22, 2, 2);
            I8_STORE_ACC(acc23, 2, 3);
            I8_STORE_ACC(acc30, 3, 0);
            I8_STORE_ACC(acc31, 3, 1);
            I8_STORE_ACC(acc32, 3, 2);
            I8_STORE_ACC(acc33, 3, 3);

#undef I8_STORE_ACC
#undef I8_UPDATE_ROW
#undef I8_INIT_ACC
        }
    }
}

void i8gemm_k_ld(const i8_t *A, const i8_t *B_reo, i32_t *C,
                 i8_t *A_reorder, const gemm_params_t *params) {
    (void)A_reorder;
    i8_sve_kernel(A, B_reo, C, params, 8, 1, 0, NULL);
}

void i8gemm_k_ld1(const i8_t *A, const i8_t *B_reo, i32_t *C,
                  i8_t *A_reorder, const gemm_params_t *params) {
    (void)A_reorder;
    i8_sve_kernel(A, B_reo, C, params, 1, 1, 0, NULL);
}

void i8gemm_k_ld2(const i8_t *A, const i8_t *B_reo, i32_t *C,
                  i8_t *A_reorder, const gemm_params_t *params) {
    (void)A_reorder;
    i8_sve_kernel(A, B_reo, C, params, 2, 1, 0, NULL);
}

void i8gemm_k_ld4(const i8_t *A, const i8_t *B_reo, i32_t *C,
                  i8_t *A_reorder, const gemm_params_t *params) {
    (void)A_reorder;
    i8_sve_kernel(A, B_reo, C, params, 4, 1, 0, NULL);
}

void i8gemm_k_ld_f(const i8_t *A, const i8_t *B_reo, f32_t *C,
                   i8_t *A_reorder, const gemm_params_t *params) {
    (void)A_reorder;
    i8_sve_kernel(A, B_reo, C, params, 8, 0, 1, NULL);
}

void i8gemm_k_ld1_f(const i8_t *A, const i8_t *B_reo, f32_t *C,
                    i8_t *A_reorder, const gemm_params_t *params) {
    (void)A_reorder;
    i8_sve_kernel(A, B_reo, C, params, 1, 0, 1, NULL);
}

void i8gemm_k_ld2_f(const i8_t *A, const i8_t *B_reo, f32_t *C,
                    i8_t *A_reorder, const gemm_params_t *params) {
    (void)A_reorder;
    i8_sve_kernel(A, B_reo, C, params, 2, 0, 1, NULL);
}

void i8gemm_k_ld4_f(const i8_t *A, const i8_t *B_reo, f32_t *C,
                    i8_t *A_reorder, const gemm_params_t *params) {
    (void)A_reorder;
    i8_sve_kernel(A, B_reo, C, params, 4, 0, 1, NULL);
}

void i8gemm_k_ld_bias_f(const i8_t *A, const i8_t *B_reo, f32_t *C,
                        i8_t *A_reorder, const gemm_params_t *params,
                        const f32_t *bias) {
    (void)A_reorder;
    i8_sve_kernel(A, B_reo, C, params, 8, 0, 1, bias);
}

void i8gemm_k_ld1_bias_f(const i8_t *A, const i8_t *B_reo, f32_t *C,
                         i8_t *A_reorder, const gemm_params_t *params,
                         const f32_t *bias) {
    (void)A_reorder;
    i8_sve_kernel(A, B_reo, C, params, 1, 0, 1, bias);
}

void i8gemm_k_ld2_bias_f(const i8_t *A, const i8_t *B_reo, f32_t *C,
                         i8_t *A_reorder, const gemm_params_t *params,
                         const f32_t *bias) {
    (void)A_reorder;
    i8_sve_kernel(A, B_reo, C, params, 2, 0, 1, bias);
}

void i8gemm_k_ld4_bias_f(const i8_t *A, const i8_t *B_reo, f32_t *C,
                         i8_t *A_reorder, const gemm_params_t *params,
                         const f32_t *bias) {
    (void)A_reorder;
    i8_sve_kernel(A, B_reo, C, params, 4, 0, 1, bias);
}

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
