// i8gemm_m16n4.c -- Experimental NEON I8MM 16x4 I8 GEMM dispatch.
//
// This path keeps the public i8gemm data contract:
//   A:     row-major int8, M x K_r
//   B_reo: packed by i8_pack_B, K_r x N_r
//   C:     row-major int32, M x N_r
//
// The micro-kernel shape follows the high-M / narrow-N flavor used by
// KleidiAI's NEON I8MM ukernel, but stores raw int32 accumulators.

#include "i8gemm.h"

#include <arm_neon.h>
#include <omp.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline int i8_min_int(int a, int b) {
    return a < b ? a : b;
}

void i8gemm_k_m16n4_packed_asm(const i8_t *A_pack, const i8_t *B_reo,
                                i32_t *C, int K_r, int N_r, int n0,
                                int rows, int zero_c);

static size_t i8_round_up_64(size_t size) {
    return (size + 63u) & ~(size_t)63u;
}

static void *i8_aligned_alloc_64(size_t size) {
    if (size == 0)
        size = 64;
    return aligned_alloc(64, i8_round_up_64(size));
}

static int i8_accumulate_c_mode(void) {
    const char *env = getenv("I8_GEMM_ACCUMULATE_C");
    return env ? atoi(env) != 0 : 0;
}

static int i8_m16n4_pack_a_mode(void) {
    const char *env = getenv("I8_M16N4_PACK_A");
    return env ? atoi(env) != 0 : 1;
}

static int i8_m16n4_use_asm_mode(void) {
    const char *env = getenv("I8_M16N4_USE_ASM");
    return env ? atoi(env) != 0 : 1;
}

static inline size_t i8_m16n4_a_block_bytes(int K_r) {
    return (size_t)16 * (size_t)K_r;
}

static void i8_pack_A_m16n4_block(const i8_t *A, i8_t *P, int rows,
                                  int K_r) {
    for (int kb = 0; kb < K_r; kb += 8) {
        for (int rp = 0; rp < 8; ++rp) {
            const int row0 = rp * 2;
            const int row1 = row0 + 1;
            int8x8_t lo = vdup_n_s8(0);
            int8x8_t hi = vdup_n_s8(0);
            if (row0 < rows)
                lo = vld1_s8(A + (size_t)row0 * (size_t)K_r + (size_t)kb);
            if (row1 < rows)
                hi = vld1_s8(A + (size_t)row1 * (size_t)K_r + (size_t)kb);
            vst1q_s8(P, vcombine_s8(lo, hi));
            P += 16;
        }
    }
}

static void i8_pack_A_m16n4(const i8_t *A, i8_t *P, int M, int K_r,
                            int num_threads) {
    const int m_blocks = (M + 15) / 16;

#pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int mb = 0; mb < m_blocks; ++mb) {
        const int m0 = mb * 16;
        const int rows = i8_min_int(16, M - m0);
        i8_pack_A_m16n4_block(A + (size_t)m0 * (size_t)K_r,
                              P + (size_t)mb * i8_m16n4_a_block_bytes(K_r),
                              rows, K_r);
    }
}

static inline int8x16_t i8_load_a_pair_packed(const i8_t *A_pack, int K_r,
                                              int rp, int kb) {
    const size_t off = (size_t)(kb >> 3) * 128u + (size_t)rp * 16u;
    (void)K_r;
    return vld1q_s8(A_pack + off);
}

static inline int8x16_t i8_load_a_pair_rowmajor(const i8_t *A, int K_r,
                                                int row0, int row1, int kb,
                                                int rows) {
    int8x8_t lo = vdup_n_s8(0);
    int8x8_t hi = vdup_n_s8(0);
    if (row0 < rows)
        lo = vld1_s8(A + (size_t)row0 * (size_t)K_r + (size_t)kb);
    if (row1 < rows)
        hi = vld1_s8(A + (size_t)row1 * (size_t)K_r + (size_t)kb);
    return vcombine_s8(lo, hi);
}

static inline int8x16_t i8_load_b_pair_k8(const i8_t *B_reo, int K_r,
                                           int n0, int kb, int col_pair) {
    const int nb8 = n0 >> 3;
    const int col = (n0 & 7) + col_pair * 2;
    const size_t off = (size_t)nb8 * (size_t)K_r * 8u +
                       (size_t)(kb >> 3) * 64u +
                       (size_t)col * 8u;
    return vld1q_s8(B_reo + off);
}

static inline void i8_store_acc_pair(i32_t *C, int ldc, int row0, int row1,
                                      int n0, int rows, int cols,
                                      int32x4_t acc) {
    if (row0 < rows) {
        if (cols > 0)
            C[(size_t)row0 * (size_t)ldc + (size_t)n0 + 0u] =
                vgetq_lane_s32(acc, 0);
        if (cols > 1)
            C[(size_t)row0 * (size_t)ldc + (size_t)n0 + 1u] =
                vgetq_lane_s32(acc, 1);
    }
    if (row1 < rows) {
        if (cols > 0)
            C[(size_t)row1 * (size_t)ldc + (size_t)n0 + 0u] =
                vgetq_lane_s32(acc, 2);
        if (cols > 1)
            C[(size_t)row1 * (size_t)ldc + (size_t)n0 + 1u] =
                vgetq_lane_s32(acc, 3);
    }
}

static void i8gemm_k_m16n4(const i8_t *A, const i8_t *A_pack,
                           const i8_t *B_reo, i32_t *C,
                           int rows, int K_r, int N_r, int n0,
                           int packed_a, int zero_c) {
    const int cols = i8_min_int(4, N_r - n0);
    int32x4_t acc[8][2];

    for (int rp = 0; rp < 8; ++rp) {
        for (int cp = 0; cp < 2; ++cp) {
            int32_t init[4] = {0, 0, 0, 0};
            const int c0 = n0 + cp * 2;
            const int row0 = rp * 2;
            const int row1 = row0 + 1;
            if (!zero_c && c0 < N_r) {
                if (row0 < rows) {
                    init[0] = C[(size_t)row0 * (size_t)N_r + (size_t)c0];
                    if (c0 + 1 < N_r)
                        init[1] = C[(size_t)row0 * (size_t)N_r + (size_t)c0 + 1u];
                }
                if (row1 < rows) {
                    init[2] = C[(size_t)row1 * (size_t)N_r + (size_t)c0];
                    if (c0 + 1 < N_r)
                        init[3] = C[(size_t)row1 * (size_t)N_r + (size_t)c0 + 1u];
                }
            }
            acc[rp][cp] = vld1q_s32(init);
        }
    }

    for (int kb = 0; kb < K_r; kb += 8) {
        const int8x16_t b0 = i8_load_b_pair_k8(B_reo, K_r, n0, kb, 0);
        const int8x16_t b1 = i8_load_b_pair_k8(B_reo, K_r, n0, kb, 1);
        for (int rp = 0; rp < 8; ++rp) {
            const int row0 = rp * 2;
            const int8x16_t a = packed_a ?
                i8_load_a_pair_packed(A_pack, K_r, rp, kb) :
                i8_load_a_pair_rowmajor(A, K_r, row0, row0 + 1, kb, rows);
            acc[rp][0] = vmmlaq_s32(acc[rp][0], a, b0);
            acc[rp][1] = vmmlaq_s32(acc[rp][1], a, b1);
        }
    }

    for (int rp = 0; rp < 8; ++rp) {
        const int row0 = rp * 2;
        i8_store_acc_pair(C, N_r, row0, row0 + 1, n0, rows,
                          i8_min_int(2, cols), acc[rp][0]);
        if (cols > 2)
            i8_store_acc_pair(C, N_r, row0, row0 + 1, n0 + 2, rows,
                              cols - 2, acc[rp][1]);
    }
}

void i8gemm_mt_dispatch_m16n4(const i8_t *A, const i8_t *B_reo,
                              i32_t *C, int M, int K_r, int N_r,
                              int num_threads) {
    if (num_threads <= 0)
        num_threads = omp_get_max_threads();
    if (num_threads <= 0)
        num_threads = 1;

    const int m_blocks = (M + 15) / 16;
    const int n_blocks = (N_r + 3) / 4;
    const int work_units = m_blocks * n_blocks;
    const int packed_a = i8_m16n4_pack_a_mode();
    const int use_asm = i8_m16n4_use_asm_mode();
    const int zero_c = !i8_accumulate_c_mode();
    if (num_threads > work_units)
        num_threads = work_units > 0 ? work_units : 1;

    i8_t *A_pack = NULL;
    if (packed_a) {
        A_pack = (i8_t *)i8_aligned_alloc_64(
            (size_t)m_blocks * i8_m16n4_a_block_bytes(K_r));
        if (!A_pack)
            return;
        i8_pack_A_m16n4(A, A_pack, M, K_r, num_threads);
    }

#pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int work = 0; work < work_units; ++work) {
        const int mb = work / n_blocks;
        const int nb = work - mb * n_blocks;
        const int m0 = mb * 16;
        const int n0 = nb * 4;
        const int rows = i8_min_int(16, M - m0);
        const i8_t *a_row = A + (size_t)m0 * (size_t)K_r;
        const i8_t *a_packed = packed_a ?
            A_pack + (size_t)mb * i8_m16n4_a_block_bytes(K_r) : NULL;
        i32_t *c_block = C + (size_t)m0 * (size_t)N_r;
        if (use_asm && packed_a && n0 + 4 <= N_r && (zero_c || rows == 16))
            i8gemm_k_m16n4_packed_asm(a_packed, B_reo, c_block,
                                       K_r, N_r, n0, rows, zero_c);
        else
            i8gemm_k_m16n4(a_row, a_packed, B_reo, c_block,
                           rows, K_r, N_r, n0, packed_a, zero_c);
    }

    free(A_pack);
}
