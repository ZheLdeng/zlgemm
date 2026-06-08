// bf16gemm_sve.h -- Public SVE BF16 GEMM API and scheduling controls.
//
// This header exposes the optimized SVE dispatcher. The default schedule is
// the current best measured policy:
//   - clamp excessive thread counts for tiny work units
//   - use N-split for large B panels with N >= 2*M
//   - keep M-split for small/H2-sized panels
//   - use the M12 body in N-split only when M >= 24 and K >= 64
//
// Link with:
//   bf16gemm_sve.c bf16gemm_sve.S -fopenmp -lm

#ifndef BF16GEMM_SVE_H
#define BF16GEMM_SVE_H

#include <stddef.h>

#include "bf16gemm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    BF16GEMM_SVE_SPLIT_AUTO = 0,
    BF16GEMM_SVE_SPLIT_M = 1,
    BF16GEMM_SVE_SPLIT_N = 2,
    BF16GEMM_SVE_SPLIT_OLD = 3
} bf16gemm_sve_split_policy_t;

typedef struct {
    bf16gemm_sve_split_policy_t split_policy;

    // 1: clamp requested threads when the shape has too little work.
    // 0: use the requested OpenMP thread count directly.
    int clamp_threads;

    // -1: auto heuristic. 0: always reorder A. >0: skip A reorder for M<=value.
    int no_reorder_max_m;

    // AUTO N-split threshold. Default is 512 KiB.
    size_t n_split_min_b_panel_bytes;

    // In N-split, enable the M12 body only above these thresholds.
    int n_split_m12_min_m;
    int n_split_m12_min_k;
} bf16gemm_sve_schedule_t;

void bf16gemm_sve_get_default_schedule(bf16gemm_sve_schedule_t *schedule);
void bf16gemm_sve_set_schedule(const bf16gemm_sve_schedule_t *schedule);
void bf16gemm_sve_get_schedule(bf16gemm_sve_schedule_t *schedule);

int bf16gemm_sve_get_n_tile(void);
int bf16gemm_sve_round_k(int K);
int bf16gemm_sve_round_n(int N);

// SVE-named aliases for the compatible BF16 API in bf16gemm.h.
void bf16gemm_sve_pack_B(const bf16_t *B, bf16_t *B_reo, int K, int N);

void bf16gemm_sve_dispatch_f32(const bf16_t *A, const bf16_t *B_reo,
                               f32_t *C, int M, int K_r, int N_r,
                               int nthreads);
void bf16gemm_sve_dispatch_bias_f32(const bf16_t *A, const bf16_t *B_reo,
                                    f32_t *C, int M, int K_r, int N_r,
                                    int nthreads, const f32_t *bias);
void bf16gemm_sve_dispatch_bf16(const bf16_t *A, const bf16_t *B_reo,
                                bf16_t *C, int M, int K_r, int N_r,
                                int nthreads);

void bf16gemm_sve_f32(const bf16_t *A_orig, const bf16_t *B_orig,
                      f32_t *C, int M, int K, int N, int nthreads);
void bf16gemm_sve_bias_f32(const bf16_t *A_orig, const bf16_t *B_orig,
                           f32_t *C, int M, int K, int N, int nthreads,
                           const f32_t *bias);
void bf16gemm_sve_bf16(const bf16_t *A_orig, const bf16_t *B_orig,
                       bf16_t *C, int M, int K, int N, int nthreads);

#ifdef __cplusplus
}
#endif

#endif // BF16GEMM_SVE_H
