// bf16gemm_sve.c -- SVE BF16 GEMM implementation with the same public API.
//
// Link this file instead of bf16gemm_mt.c/bf16gemm_k*.S when building the
// SVE variant. The exported function names intentionally match the NEON path.

#include <arm_sve.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "bf16gemm_sve.h"
#include "gemm_params.h"

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

static bf16gemm_sve_schedule_t bf16_sve_schedule = {
    BF16GEMM_SVE_SPLIT_AUTO,
    0,
    -1,
    512u * 1024u,
    24,
    64
};

void bf16gemm_sve_get_default_schedule(bf16gemm_sve_schedule_t *schedule) {
    if (!schedule)
        return;
    schedule->split_policy = BF16GEMM_SVE_SPLIT_AUTO;
    schedule->clamp_threads = 0;
    schedule->no_reorder_max_m = -1;
    schedule->n_split_min_b_panel_bytes = 512u * 1024u;
    schedule->n_split_m12_min_m = 24;
    schedule->n_split_m12_min_k = 64;
}

void bf16gemm_sve_set_schedule(const bf16gemm_sve_schedule_t *schedule) {
    if (!schedule) {
        bf16gemm_sve_get_default_schedule(&bf16_sve_schedule);
        return;
    }

    bf16_sve_schedule = *schedule;
    if (bf16_sve_schedule.split_policy < BF16GEMM_SVE_SPLIT_AUTO ||
        bf16_sve_schedule.split_policy > BF16GEMM_SVE_SPLIT_OLD)
        bf16_sve_schedule.split_policy = BF16GEMM_SVE_SPLIT_AUTO;
    bf16_sve_schedule.clamp_threads = bf16_sve_schedule.clamp_threads != 0;
    if (bf16_sve_schedule.no_reorder_max_m < -1)
        bf16_sve_schedule.no_reorder_max_m = -1;
    if (bf16_sve_schedule.n_split_min_b_panel_bytes == 0)
        bf16_sve_schedule.n_split_min_b_panel_bytes = 512u * 1024u;
    if (bf16_sve_schedule.n_split_m12_min_m < 0)
        bf16_sve_schedule.n_split_m12_min_m = 0;
    if (bf16_sve_schedule.n_split_m12_min_k < 0)
        bf16_sve_schedule.n_split_m12_min_k = 0;
}

void bf16gemm_sve_get_schedule(bf16gemm_sve_schedule_t *schedule) {
    if (schedule)
        *schedule = bf16_sve_schedule;
}

int bf16gemm_sve_get_n_tile(void) {
    return bf16_sve_n_tile();
}

int bf16gemm_sve_round_k(int K) {
    return round_up_int(K < 8 ? 8 : K, 8);
}

int bf16gemm_sve_round_n(int N) {
    return round_up_int(N < 8 ? 8 : N, bf16_sve_n_tile());
}

static int bf16_no_reorder_max_m_env(int *has_env) {
    static int value = -1;
    static int present = -1;
    if (value < 0) {
        const char *env = getenv("BF16_SVE_NOREORDER_MAX_M");
        present = env != NULL;
        value = present ? atoi(env) : 0;
        if (value < 0)
            value = 0;
    }
    if (has_env)
        *has_env = present;
    return value;
}

static int bf16_use_no_reorder_for_shape(int M, int K_r, int n_tiles) {
    int has_env = 0;
    int max_m = bf16_sve_schedule.no_reorder_max_m;
    int env_max_m = bf16_no_reorder_max_m_env(&has_env);
    if (has_env)
        max_m = env_max_m;
    if (max_m >= 0)
        return max_m > 0 && M <= max_m;
    if (M <= 8 && n_tiles <= 2)
        return 1;
    /*
     * Small-M, larger-N: skip the separate A-pack OpenMP region (no_reorder) so
     * a tiny shape does not fork 64-80 threads to pack a few rows and collapse.
     * Mirrors the i8 small-M -> hybrid routing (M<=16, K_r<=1024) that fixed the
     * identical i8 8x512x512 collapse. Measured on Huawei (bf16, t64): without
     * this 8x512x512 runs the A-pack fork and collapses to ~72 GFLOPS; the
     * no-reorder path scales (~363+). K bounded so deep-K small-M (already fine
     * on the packed path, e.g. 8x4096x1024) is unaffected.
     */
    return M <= 16 && K_r <= 1024;
}

static int bf16_effective_threads(int num_threads, int M, int K_r, int N_r,
                                  int n_tiles) {
    if (num_threads <= 0)
        num_threads = omp_get_max_threads();
    if (num_threads <= 0)
        num_threads = 1;
    int clamp_threads = bf16_sve_schedule.clamp_threads;
    const char *clamp_env = getenv("BF16_SVE_CLAMP_THREADS");
    if (clamp_env)
        clamp_threads = atoi(clamp_env) != 0;
    if (!clamp_threads)
        return num_threads;

    const int m_blocks = (M + 7) / 8;
    int work_units = m_blocks * n_tiles;
    if (work_units < 1)
        work_units = 1;
    if (num_threads > work_units)
        num_threads = work_units;

    /*
     * Very small N-tile counts do not have enough independent work to amortize
     * an OpenMP team. Keep at least four N tiles per thread for sub-64-row
     * shapes; larger shapes still benefit from M splitting.
     */
    if (M < 64 && n_tiles < num_threads * 4) {
        int by_n = (n_tiles + 3) / 4;
        if (by_n < 1)
            by_n = 1;
        if (num_threads > by_n)
            num_threads = by_n;
    }

    return num_threads;
}

static int bf16_use_n_split(int M, int K_r, int N_r, int n_tiles,
                            int num_threads) {
    const char *split = getenv("BF16_SVE_SPLIT");
    if (split) {
        if (strcmp(split, "m") == 0)
            return 0;
        if (strcmp(split, "n") == 0)
            return n_tiles >= num_threads;
        if (strcmp(split, "old") == 0)
            return !(M / 8 >= num_threads || n_tiles < num_threads);
    }

    switch (bf16_sve_schedule.split_policy) {
    case BF16GEMM_SVE_SPLIT_M:
        return 0;
    case BF16GEMM_SVE_SPLIT_N:
        return n_tiles >= num_threads;
    case BF16GEMM_SVE_SPLIT_OLD:
        return !(M / 8 >= num_threads || n_tiles < num_threads);
    case BF16GEMM_SVE_SPLIT_AUTO:
    default:
        break;
    }

    if (num_threads <= 1)
        return 0;
    if (n_tiles < num_threads)
        return 0;
    if (M / 8 < num_threads)
        return 1;

    /*
     * M splitting duplicates the whole B panel across threads. Prefer N
     * splitting when B is clearly wider than A, because it partitions B and
     * only duplicates the usually smaller packed-A stream.
     */
    const size_t b_panel_bytes = (size_t)K_r * (size_t)N_r * sizeof(bf16_t);
    return b_panel_bytes >= bf16_sve_schedule.n_split_min_b_panel_bytes &&
           N_r >= M * 2;
}

static int bf16_clamp_threads_for_split(int num_threads, int M, int n_tiles,
                                        int use_n_split) {
    int clamp_threads = bf16_sve_schedule.clamp_threads;
    const char *clamp_env = getenv("BF16_SVE_CLAMP_THREADS");
    if (clamp_env)
        clamp_threads = atoi(clamp_env) != 0;
    if (!clamp_threads || num_threads <= 1)
        return num_threads;

    /*
     * Opt-in clamp for machines where one work unit per active thread is too
     * thin for OpenMP overhead. Keep at least two M8 blocks per thread for
     * M-split and at least two N tiles per thread for N-split. Forced and
     * experimental block split paths keep their own policy.
     */
    int max_threads;
    if (use_n_split) {
        max_threads = (n_tiles + 1) / 2;
    } else {
        const int m_blocks = (M + 7) / 8;
        max_threads = (m_blocks + 1) / 2;
    }
    if (max_threads < 1)
        max_threads = 1;
    if (num_threads > max_threads)
        num_threads = max_threads;
    return num_threads;
}

static int bf16_use_2d_split(int M, int K_r, int N_r, int n_tiles,
                             int num_threads) {
    const char *split = getenv("BF16_SVE_SPLIT");
    if (split)
        return strcmp(split, "2d") == 0;

    if (num_threads < 10 || K_r < 512)
        return 0;

    /*
     * v2 80c data shows SVE BF16 benefits from 2D splitting once there are
     * enough M and N work units to keep high thread counts busy.  The guard
     * avoids using 2D for tiny grids where extra task geometry costs more than
     * it saves.
     */
    const int m_units = (M + 7) / 8;
    if (m_units < 2 || n_tiles < 1)
        return 0;
    /*
     * Narrow-N guard (mirrors the i8 n_tiles>=8 fix). With very few N tiles the
     * 2D (m_unit x n_tile) collapse loses badly to a plain M-split -- measured
     * on Huawei 80c: 2048x16384x24 (n_tiles=2) @t64 collapse=795 vs M-split=1465
     * (+85%), @t48 +58%, @t32 +36%. At n_tiles>=4 the collapse is fine
     * (2048x4096x64 @t32 2D=1865 > M-split). So require >=4 N tiles for 2D;
     * narrower N falls back to the M-split, which streams B per row-band.
     */
    if (n_tiles < 4)
        return 0;
    if ((size_t)m_units * (size_t)n_tiles < (size_t)num_threads * 2)
        return 0;
    return M >= 64 || N_r >= 64;
}

static int bf16_use_ngroup_split(void) {
    const char *split = getenv("BF16_SVE_SPLIT");
    return split && strcmp(split, "ngroup") == 0;
}

static int bf16_use_auto_nblock_split(int M, int K_r, int N_r,
                                      int num_threads) {
    if (num_threads != 1)
        return 0;
    const char *split = getenv("BF16_SVE_SPLIT");
    if (split)
        return strcmp(split, "nblock") == 0;
    return M >= 512 && K_r >= 2048 && N_r >= 8192;
}

static int bf16_use_nblock_mt_split(int M, int K_r, int N_r,
                                    int num_threads) {
    const char *split = getenv("BF16_SVE_SPLIT");
    if (num_threads <= 1)
        return 0;
    if (split)
        return strcmp(split, "nblock") == 0;
    return M >= 512 && K_r >= 2048 && N_r >= 8192;
}

static int bf16_use_kblock_split(void) {
    const char *split = getenv("BF16_SVE_SPLIT");
    return split && strcmp(split, "kblock") == 0;
}

static int bf16_2d_order_m_first(void) {
    const char *order = getenv("BF16_SVE_2D_ORDER");
    return order && strcmp(order, "m") == 0;
}

static int bf16_ngroup_count(int n_tiles, int num_threads) {
    int groups = 0;
    const char *env = getenv("BF16_SVE_N_GROUPS");
    if (env)
        groups = atoi(env);
    if (groups <= 0)
        groups = num_threads;
    if (groups > num_threads)
        groups = num_threads;
    if (groups > n_tiles)
        groups = n_tiles;
    if (groups < 1)
        groups = 1;
    return groups;
}

static int bf16_nblock_tiles(int n_tiles, int K_r, int num_threads) {
    int block = 0;
    const char *env = getenv("BF16_SVE_NB_TILES");
    if (env)
        block = atoi(env);
    if (block <= 0) {
        if (num_threads > 1 && K_r >= 4096)
            block = 4;
        else
            block = 8;
    }
    if (block > n_tiles)
        block = n_tiles;
    if (block < 1)
        block = 1;
    return block;
}

static int bf16_kblock_size(int K_r) {
    int block = 0;
    const char *env = getenv("BF16_SVE_K_BLOCK");
    if (env)
        block = atoi(env);
    if (block <= 0)
        block = 256;
    block = (block / 8) * 8;
    if (block < 8)
        block = 8;
    if (block > K_r)
        block = K_r;
    return block;
}

static void bf16_thread_group(int tid, int nthreads, int groups,
                              int *group, int *local_tid,
                              int *local_threads) {
    int base = nthreads / groups;
    int extra = nthreads % groups;
    int wide_threads = (base + 1) * extra;
    if (tid < wide_threads) {
        *group = tid / (base + 1);
        *local_tid = tid - *group * (base + 1);
        *local_threads = base + 1;
    } else {
        int off = tid - wide_threads;
        *group = extra + off / base;
        *local_tid = off - (off / base) * base;
        *local_threads = base;
    }
}

static size_t bf16_a_reorder_stride(int K_r) {
    return (size_t)K_r * 8;
}

static size_t bf16_max_size(size_t a, size_t b) {
    return a > b ? a : b;
}

static void bf16_static_range(int total, int tid, int nthreads,
                              int *start, int *end) {
    int blocks_per = total / nthreads;
    int extra = total % nthreads;
    if (tid < extra) {
        *start = tid * (blocks_per + 1);
        *end = *start + blocks_per + 1;
    } else {
        *start = extra * (blocks_per + 1) + (tid - extra) * blocks_per;
        *end = *start + blocks_per;
    }
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

static void bf16_pack_A_sve_block_lda(const bf16_t *A, bf16_t *A_reo,
                                      int M, int K_count, int lda) {
    int rowpairs = (M + 1) / 2;
    if (rowpairs < 4)
        rowpairs = 4;
    if (rowpairs > 6)
        rowpairs = 6;
    size_t idx = 0;

    for (int kb = 0; kb < K_count; kb += 4) {
        for (int rp = 0; rp < rowpairs; rp++) {
            int r0 = rp * 2;
            int r1 = r0 + 1;
            for (int k = 0; k < 4; k++)
                A_reo[idx++] = (r0 < M) ? A[(size_t)r0 * lda + kb + k] : 0;
            for (int k = 0; k < 4; k++)
                A_reo[idx++] = (r1 < M) ? A[(size_t)r1 * lda + kb + k] : 0;
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
void bf16gemm_k_nld_f(const bf16_t *A, const bf16_t *B_reo, f32_t *C,
                      bf16_t *A_reorder, const gemm_params_t *params);
void bf16gemm_k_nld1_f(const bf16_t *A, const bf16_t *B_reo, f32_t *C,
                       bf16_t *A_reorder, const gemm_params_t *params);
void bf16gemm_k_nld2_f(const bf16_t *A, const bf16_t *B_reo, f32_t *C,
                       bf16_t *A_reorder, const gemm_params_t *params);
void bf16gemm_k_nld4_f(const bf16_t *A, const bf16_t *B_reo, f32_t *C,
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
void bf16gemm_k_ld_f_m12(const bf16_t *A, const bf16_t *B_reo, f32_t *C,
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
        if (no_load) {
            if (M <= 1)
                bf16gemm_k_nld1_f(A, B_reo, C, A_reorder, &p);
            else if (M <= 2)
                bf16gemm_k_nld2_f(A, B_reo, C, A_reorder, &p);
            else if (M <= 4)
                bf16gemm_k_nld4_f(A, B_reo, C, A_reorder, &p);
            else
                bf16gemm_k_nld_f(A, B_reo, C, A_reorder, &p);
        } else {
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
    const int n_tile = bf16_sve_n_tile();
    const int n_tiles = N_r / n_tile;
    num_threads = bf16_effective_threads(num_threads, M, K_r, N_r, n_tiles);
    const size_t a_stride = bf16_a_reorder_stride(K_r);
    const int no_reorder = bf16_use_no_reorder_for_shape(M, K_r, n_tiles);
    const int use_n_split = bf16_use_n_split(M, K_r, N_r, n_tiles, num_threads);

    if (bf16_use_kblock_split() && num_threads == 1 &&
        ((M % 12) == 0 || (M % 8) == 0)) {
        int k_block = bf16_kblock_size(K_r);
        int mr = (M % 12) == 0 ? 12 : 8;
        bf16_t *A_reo = (bf16_t *)aligned_alloc_64(
            (size_t)k_block * (size_t)mr * sizeof(bf16_t));
        if (A_reo) {
            for (int m0 = 0; m0 < M; m0 += mr) {
                for (int kb = 0; kb < K_r; kb += k_block) {
                    int kc = K_r - kb;
                    if (kc > k_block)
                        kc = k_block;
                    bf16_pack_A_sve_block_lda(A + (size_t)m0 * K_r + kb,
                                              A_reo, mr, kc, K_r);
                    gemm_params_t p = {mr, kc, N_r, K_r, K_r, N_r};
                    const bf16_t *B_k = B_reo + (size_t)kb * n_tile;
                    f32_t *C_m = C + (size_t)m0 * N_r;
                    if (mr == 12) {
                        if (kb == 0)
                            bf16gemm_k_nld_f_m12(A + (size_t)m0 * K_r + kb,
                                                 B_k, C_m, A_reo, &p);
                        else
                            bf16gemm_k_ld_f_m12(A + (size_t)m0 * K_r + kb,
                                                B_k, C_m, A_reo, &p);
                    } else {
                        if (kb == 0)
                            bf16gemm_k_nld_f(A + (size_t)m0 * K_r + kb,
                                             B_k, C_m, A_reo, &p);
                        else
                            bf16gemm_k_ld(A + (size_t)m0 * K_r + kb,
                                          B_k, C_m, A_reo, &p);
                    }
                }
            }
            free(A_reo);
            return;
        }
    }

    if (bf16_use_auto_nblock_split(M, K_r, N_r, num_threads)) {
        int blocks12 = (!no_reorder &&
                        M >= bf16_sve_schedule.n_split_m12_min_m &&
                        K_r >= bf16_sve_schedule.n_split_m12_min_k) ?
                       M / 12 : 0;
        int m12_end = blocks12 * 12;
        const size_t a12_stride = (size_t)K_r * 12;
        bf16_t *A_reo12_pool = blocks12 > 0 ?
            (bf16_t *)aligned_alloc_64((size_t)blocks12 * a12_stride *
                                       sizeof(bf16_t)) : NULL;
        if (!A_reo12_pool) {
            blocks12 = 0;
            m12_end = 0;
        }

        int tail_M = M - m12_end;
        int tail_blocks = (tail_M + 7) / 8;
        bf16_t *A_reo_pool = (!no_reorder && tail_blocks > 0) ?
            (bf16_t *)aligned_alloc_64((size_t)tail_blocks * a_stride *
                                       sizeof(bf16_t)) : NULL;
        int nb_tiles = bf16_nblock_tiles(n_tiles, K_r, num_threads);

        if (A_reo12_pool) {
            for (int b = 0; b < blocks12; b++)
                bf16_pack_A_sve_block(A + (size_t)b * 12 * K_r,
                                      A_reo12_pool + (size_t)b * a12_stride,
                                      12, K_r);
        }
        if (A_reo_pool) {
            for (int b = 0; b < tail_blocks; b++) {
                int m0 = m12_end + b * 8;
                int mb = M - m0 < 8 ? M - m0 : 8;
                bf16_pack_A_sve_block(A + (size_t)m0 * K_r,
                                      A_reo_pool + (size_t)b * a_stride,
                                      mb, K_r);
            }
        }

        for (int nb0 = 0; nb0 < n_tiles; nb0 += nb_tiles) {
            int nb_end = nb0 + nb_tiles;
            if (nb_end > n_tiles)
                nb_end = n_tiles;
            int n0 = nb0 * n_tile;
            int nb_cols = (nb_end - nb0) * n_tile;

            if (A_reo12_pool) {
                for (int b = 0; b < blocks12; b++) {
                    int m0 = b * 12;
                    gemm_params_t p = {12, K_r, nb_cols, K_r, K_r, N_r};
                    bf16gemm_k_nld_f_m12(A + (size_t)m0 * K_r,
                                         B_reo + (size_t)nb0 * K_r * n_tile,
                                         C + (size_t)m0 * N_r + n0,
                                         A_reo12_pool + (size_t)b * a12_stride,
                                         &p);
                }
            }

            for (int b = 0; b < tail_blocks; b++) {
                int m0 = m12_end + b * 8;
                int mb = M - m0 < 8 ? M - m0 : 8;
                bf16_t *A_reo = A_reo_pool ?
                    A_reo_pool + (size_t)b * a_stride : NULL;
                bf16_dispatch_f32(A + (size_t)m0 * K_r,
                                  B_reo + (size_t)nb0 * K_r * n_tile,
                                  C + (size_t)m0 * N_r + n0,
                                  mb, K_r, nb_cols, N_r, 1, NULL, A_reo);
            }
        }

        free(A_reo_pool);
        free(A_reo12_pool);
        return;
    }

    if (bf16_use_nblock_mt_split(M, K_r, N_r, num_threads)) {
        int blocks12 = (!no_reorder &&
                        M >= bf16_sve_schedule.n_split_m12_min_m &&
                        K_r >= bf16_sve_schedule.n_split_m12_min_k) ?
                       M / 12 : 0;
        int m12_end = blocks12 * 12;
        const size_t a12_stride = (size_t)K_r * 12;
        bf16_t *A_reo12_pool = blocks12 > 0 ?
            (bf16_t *)aligned_alloc_64((size_t)blocks12 * a12_stride *
                                       sizeof(bf16_t)) : NULL;
        if (!A_reo12_pool) {
            blocks12 = 0;
            m12_end = 0;
        }

        const int tail_M = M - m12_end;
        const int tail_blocks = (tail_M + 7) / 8;
        bf16_t *A_reo_pool = (!no_reorder && tail_blocks > 0) ?
            (bf16_t *)aligned_alloc_64((size_t)tail_blocks * a_stride *
                                       sizeof(bf16_t)) : NULL;
        const int nb_tiles = bf16_nblock_tiles(n_tiles, K_r, num_threads);

        #pragma omp parallel num_threads(num_threads)
        {
            if (A_reo12_pool) {
                #pragma omp for schedule(static)
                for (int b = 0; b < blocks12; b++)
                    bf16_pack_A_sve_block(A + (size_t)b * 12 * K_r,
                                          A_reo12_pool + (size_t)b * a12_stride,
                                          12, K_r);
            }

            if (A_reo_pool) {
                #pragma omp for schedule(static)
                for (int b = 0; b < tail_blocks; b++) {
                    int m0 = m12_end + b * 8;
                    int mb = M - m0 < 8 ? M - m0 : 8;
                    bf16_pack_A_sve_block(A + (size_t)m0 * K_r,
                                          A_reo_pool + (size_t)b * a_stride,
                                          mb, K_r);
                }
            }

            #pragma omp for schedule(static)
            for (int nb0 = 0; nb0 < n_tiles; nb0 += nb_tiles) {
                int nb_end = nb0 + nb_tiles;
                if (nb_end > n_tiles)
                    nb_end = n_tiles;
                int n0 = nb0 * n_tile;
                int nb_cols = (nb_end - nb0) * n_tile;

                if (A_reo12_pool) {
                    for (int b = 0; b < blocks12; b++) {
                        int m0 = b * 12;
                        gemm_params_t p = {12, K_r, nb_cols, K_r, K_r, N_r};
                        bf16gemm_k_nld_f_m12(A + (size_t)m0 * K_r,
                                             B_reo + (size_t)nb0 * K_r * n_tile,
                                             C + (size_t)m0 * N_r + n0,
                                             A_reo12_pool + (size_t)b * a12_stride,
                                             &p);
                    }
                }

                for (int b = 0; b < tail_blocks; b++) {
                    int m0 = m12_end + b * 8;
                    int mb = M - m0 < 8 ? M - m0 : 8;
                    bf16_t *A_reo = A_reo_pool ?
                        A_reo_pool + (size_t)b * a_stride : NULL;
                    bf16_dispatch_f32(A + (size_t)m0 * K_r,
                                      B_reo + (size_t)nb0 * K_r * n_tile,
                                      C + (size_t)m0 * N_r + n0,
                                      mb, K_r, nb_cols, N_r, 1, NULL, A_reo);
                }
            }
        }

        free(A_reo_pool);
        free(A_reo12_pool);
        return;
    }

    if (bf16_use_ngroup_split() && num_threads > 1) {
        int blocks12 = (!no_reorder &&
                        M >= bf16_sve_schedule.n_split_m12_min_m &&
                        K_r >= bf16_sve_schedule.n_split_m12_min_k) ?
                       M / 12 : 0;
        int m12_end = blocks12 * 12;
        const size_t a12_stride = (size_t)K_r * 12;
        bf16_t *A_reo12_pool = blocks12 > 0 ?
            (bf16_t *)aligned_alloc_64((size_t)blocks12 * a12_stride *
                                       sizeof(bf16_t)) : NULL;
        if (!A_reo12_pool) {
            blocks12 = 0;
            m12_end = 0;
        }

        const int tail_M = M - m12_end;
        const int tail_blocks = (tail_M + 7) / 8;
        bf16_t *A_reo_pool = (!no_reorder && tail_blocks > 0) ?
            (bf16_t *)aligned_alloc_64((size_t)tail_blocks * a_stride *
                                       sizeof(bf16_t)) : NULL;
        const int total_m_units = blocks12 + tail_blocks;
        const int requested_groups = bf16_ngroup_count(n_tiles, num_threads);

        #pragma omp parallel num_threads(num_threads)
        {
            if (A_reo12_pool) {
                #pragma omp for schedule(static)
                for (int b = 0; b < blocks12; b++)
                    bf16_pack_A_sve_block(A + (size_t)b * 12 * K_r,
                                          A_reo12_pool + (size_t)b * a12_stride,
                                          12, K_r);
            }

            if (A_reo_pool) {
                #pragma omp for schedule(static)
                for (int b = 0; b < tail_blocks; b++) {
                    int m0 = m12_end + b * 8;
                    int mb = M - m0 < 8 ? M - m0 : 8;
                    bf16_pack_A_sve_block(A + (size_t)m0 * K_r,
                                          A_reo_pool + (size_t)b * a_stride,
                                          mb, K_r);
                }
            }

            int tid = omp_get_thread_num();
            int nth = omp_get_num_threads();
            int groups = requested_groups <= nth ? requested_groups : nth;
            int group = 0;
            int local_tid = 0;
            int local_threads = 1;
            int n_start = 0;
            int n_end = 0;
            int m_start = 0;
            int m_end = 0;

            bf16_thread_group(tid, nth, groups, &group, &local_tid,
                              &local_threads);
            bf16_static_range(n_tiles, group, groups, &n_start, &n_end);
            bf16_static_range(total_m_units, local_tid, local_threads,
                              &m_start, &m_end);

            for (int t = n_start; t < n_end; t++) {
                int n0 = t * n_tile;
                for (int m_unit = m_start; m_unit < m_end; m_unit++) {
                    if (m_unit < blocks12) {
                        int m0 = m_unit * 12;
                        gemm_params_t p = {12, K_r, n_tile, K_r, K_r, N_r};
                        bf16gemm_k_nld_f_m12(A + (size_t)m0 * K_r,
                                             B_reo + (size_t)t * K_r * n_tile,
                                             C + (size_t)m0 * N_r + n0,
                                             A_reo12_pool + (size_t)m_unit * a12_stride,
                                             &p);
                    } else {
                        int tb = m_unit - blocks12;
                        int m0 = m12_end + tb * 8;
                        int mb = M - m0 < 8 ? M - m0 : 8;
                        bf16_t *A_reo = A_reo_pool ?
                            A_reo_pool + (size_t)tb * a_stride : NULL;
                        bf16_dispatch_f32(A + (size_t)m0 * K_r,
                                          B_reo + (size_t)t * K_r * n_tile,
                                          C + (size_t)m0 * N_r + n0,
                                          mb, K_r, n_tile, N_r, 1, NULL, A_reo);
                    }
                }
            }
        }

        free(A_reo_pool);
        free(A_reo12_pool);
        return;
    }

    if (bf16_use_2d_split(M, K_r, N_r, n_tiles, num_threads) &&
        num_threads > 1) {
        int blocks12 = (!no_reorder &&
                        M >= bf16_sve_schedule.n_split_m12_min_m &&
                        K_r >= bf16_sve_schedule.n_split_m12_min_k) ?
                       M / 12 : 0;
        int m12_end = blocks12 * 12;
        const size_t a12_stride = (size_t)K_r * 12;
        bf16_t *A_reo12_pool = blocks12 > 0 ?
            (bf16_t *)aligned_alloc_64((size_t)blocks12 * a12_stride *
                                       sizeof(bf16_t)) : NULL;
        if (!A_reo12_pool) {
            blocks12 = 0;
            m12_end = 0;
        }

        const int tail_M = M - m12_end;
        const int tail_blocks = (tail_M + 7) / 8;
        bf16_t *A_reo_pool = (!no_reorder && tail_blocks > 0) ?
            (bf16_t *)aligned_alloc_64((size_t)tail_blocks * a_stride *
                                       sizeof(bf16_t)) : NULL;
        const int total_m_units = blocks12 + tail_blocks;
        const int m_first = bf16_2d_order_m_first();

        #pragma omp parallel num_threads(num_threads)
        {
            if (A_reo12_pool) {
                #pragma omp for schedule(static)
                for (int b = 0; b < blocks12; b++)
                    bf16_pack_A_sve_block(A + (size_t)b * 12 * K_r,
                                          A_reo12_pool + (size_t)b * a12_stride,
                                          12, K_r);
            }

            if (A_reo_pool) {
                #pragma omp for schedule(static)
                for (int b = 0; b < tail_blocks; b++) {
                    int m0 = m12_end + b * 8;
                    int mb = M - m0 < 8 ? M - m0 : 8;
                    bf16_pack_A_sve_block(A + (size_t)m0 * K_r,
                                          A_reo_pool + (size_t)b * a_stride,
                                          mb, K_r);
                }
            }

            #pragma omp for schedule(static)
            for (int work = 0; work < total_m_units * n_tiles; work++) {
                int m_unit;
                int t;
                if (m_first) {
                    m_unit = work / n_tiles;
                    t = work - m_unit * n_tiles;
                } else {
                    t = work / total_m_units;
                    m_unit = work - t * total_m_units;
                }

                int n0 = t * n_tile;
                if (m_unit < blocks12) {
                    int m0 = m_unit * 12;
                    gemm_params_t p = {12, K_r, n_tile, K_r, K_r, N_r};
                    bf16gemm_k_nld_f_m12(A + (size_t)m0 * K_r,
                                         B_reo + (size_t)t * K_r * n_tile,
                                         C + (size_t)m0 * N_r + n0,
                                         A_reo12_pool + (size_t)m_unit * a12_stride,
                                         &p);
                } else {
                    int tb = m_unit - blocks12;
                    int m0 = m12_end + tb * 8;
                    int mb = M - m0 < 8 ? M - m0 : 8;
                    bf16_t *A_reo = A_reo_pool ?
                        A_reo_pool + (size_t)tb * a_stride : NULL;
                    bf16_dispatch_f32(A + (size_t)m0 * K_r,
                                      B_reo + (size_t)t * K_r * n_tile,
                                      C + (size_t)m0 * N_r + n0,
                                      mb, K_r, n_tile, N_r, 1, NULL, A_reo);
                }
            }
        }

        free(A_reo_pool);
        free(A_reo12_pool);
        return;
    }

    num_threads = bf16_clamp_threads_for_split(num_threads, M, n_tiles,
                                               use_n_split);

    if (!use_n_split) {
        if (no_reorder) {
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            for (int m0 = 0; m0 < M; m0 += 8) {
                int mb = M - m0 < 8 ? M - m0 : 8;
                bf16_dispatch_f32(A + (size_t)m0 * K_r, B_reo,
                                  C + (size_t)m0 * N_r, mb, K_r, N_r, N_r,
                                  1, NULL, NULL);
            }
            return;
        }

        const int blocks12 = M / 12;
        const int m12_end = blocks12 * 12;
        const int tail_M = M - m12_end;
        const int tail_blocks = (tail_M + 7) / 8;
        const size_t a12_stride = (size_t)K_r * 12;
        const size_t scratch_stride = bf16_max_size(a12_stride, a_stride);
        bf16_t *scratch_pool = (bf16_t *)aligned_alloc_64(
            (size_t)num_threads * scratch_stride * sizeof(bf16_t));

        if (scratch_pool) {
            #pragma omp parallel num_threads(num_threads)
            {
                int tid = omp_get_thread_num();
                int nth = omp_get_num_threads();
                int start = 0;
                int end = 0;
                bf16_t *scratch = scratch_pool + (size_t)tid * scratch_stride;

                bf16_static_range(blocks12, tid, nth, &start, &end);
                for (int b = start; b < end; b++) {
                    int m0 = b * 12;
                    gemm_params_t p = {12, K_r, N_r, K_r, K_r, N_r};
                    bf16_pack_A_sve_block(A + (size_t)m0 * K_r, scratch, 12, K_r);
                    bf16gemm_k_nld_f_m12(A + (size_t)m0 * K_r, B_reo,
                                         C + (size_t)m0 * N_r, scratch, &p);
                }

                bf16_static_range(tail_blocks, tid, nth, &start, &end);
                for (int b = start; b < end; b++) {
                    int m0 = m12_end + b * 8;
                    int mb = M - m0 < 8 ? M - m0 : 8;
                    bf16_pack_A_sve_block(A + (size_t)m0 * K_r, scratch, mb, K_r);
                    bf16_dispatch_f32(A + (size_t)m0 * K_r, B_reo,
                                      C + (size_t)m0 * N_r, mb, K_r, N_r,
                                      N_r, 1, NULL, scratch);
                }
            }
            free(scratch_pool);
        } else {
            bf16_t *A_reo12_pool =
                bf16_prepare_A_reorder_pool_m12(A, blocks12, K_r, num_threads);
            if (A_reo12_pool) {
                #pragma omp parallel for num_threads(num_threads) schedule(static)
                for (int b = 0; b < blocks12; b++) {
                    int m0 = b * 12;
                    gemm_params_t p = {12, K_r, N_r, K_r, K_r, N_r};
                    bf16gemm_k_nld_f_m12(A + (size_t)m0 * K_r, B_reo,
                                         C + (size_t)m0 * N_r,
                                         A_reo12_pool + (size_t)b * a12_stride,
                                         &p);
                }
            }

            bf16_t *A_reo_pool =
                bf16_prepare_A_reorder_pool(A + (size_t)m12_end * K_r,
                                            tail_M, K_r, num_threads);
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            for (int m0 = m12_end; m0 < M; m0 += 8) {
                int mb = M - m0 < 8 ? M - m0 : 8;
                bf16_t *A_reo = A_reo_pool ?
                    A_reo_pool + (size_t)((m0 - m12_end) / 8) * a_stride : NULL;
                bf16_dispatch_f32(A + (size_t)m0 * K_r, B_reo,
                                  C + (size_t)m0 * N_r, mb, K_r, N_r, N_r,
                                  1, NULL, A_reo);
            }
            free(A_reo_pool);
            free(A_reo12_pool);
        }
    } else {
        int blocks12 = (!no_reorder &&
                        M >= bf16_sve_schedule.n_split_m12_min_m &&
                        K_r >= bf16_sve_schedule.n_split_m12_min_k) ?
                       M / 12 : 0;
        int m12_end = blocks12 * 12;
        bf16_t *A_reo12_pool = blocks12 > 0 ?
            (bf16_t *)aligned_alloc_64((size_t)blocks12 * K_r * 12 *
                                       sizeof(bf16_t)) : NULL;
        if (!A_reo12_pool) {
            blocks12 = 0;
            m12_end = 0;
        }
        int tail_M = M - m12_end;
        int tail_blocks = (tail_M + 7) / 8;
        bf16_t *A_reo_pool = (!no_reorder && tail_blocks > 0) ?
            (bf16_t *)aligned_alloc_64((size_t)tail_blocks * a_stride *
                                       sizeof(bf16_t)) : NULL;

        #pragma omp parallel num_threads(num_threads)
        {
            if (A_reo12_pool) {
                const size_t a12_stride = (size_t)K_r * 12;
                #pragma omp for schedule(static)
                for (int b = 0; b < blocks12; b++)
                    bf16_pack_A_sve_block(A + (size_t)b * 12 * K_r,
                                          A_reo12_pool + (size_t)b * a12_stride,
                                          12, K_r);
            }

            if (A_reo_pool) {
                #pragma omp for schedule(static)
                for (int b = 0; b < tail_blocks; b++) {
                    int m0 = m12_end + b * 8;
                    int mb = M - m0 < 8 ? M - m0 : 8;
                    bf16_pack_A_sve_block(A + (size_t)m0 * K_r,
                                          A_reo_pool + (size_t)b * a_stride,
                                          mb, K_r);
                }
            }

            #pragma omp for schedule(static)
            for (int t = 0; t < n_tiles; t++) {
                int n0 = t * n_tile;
                if (A_reo12_pool) {
                    const size_t a12_stride = (size_t)K_r * 12;
                    for (int b = 0; b < blocks12; b++) {
                        gemm_params_t p = {12, K_r, n_tile, K_r, K_r, N_r};
                        int m0 = b * 12;
                        bf16gemm_k_nld_f_m12(A + (size_t)m0 * K_r,
                                             B_reo + (size_t)t * K_r * n_tile,
                                             C + (size_t)m0 * N_r + n0,
                                             A_reo12_pool + (size_t)b * a12_stride,
                                             &p);
                    }
                }
                for (int m0 = m12_end; m0 < M; m0 += 8) {
                    int mb = M - m0 < 8 ? M - m0 : 8;
                    bf16_t *A_reo = A_reo_pool ?
                        A_reo_pool + (size_t)((m0 - m12_end) / 8) * a_stride : NULL;
                    bf16_dispatch_f32(A + (size_t)m0 * K_r,
                                      B_reo + (size_t)t * K_r * n_tile,
                                      C + (size_t)m0 * N_r + n0,
                                      mb, K_r, n_tile, N_r, 1, NULL, A_reo);
                }
            }
        }
        free(A_reo_pool);
        free(A_reo12_pool);
    }
}

void bf16gemm_mt_dispatch_bias_f(const bf16_t *A, const bf16_t *B_reo,
                                 f32_t *C, int M, int K_r, int N_r,
                                 int num_threads, const f32_t *bias) {
    const int n_tile = bf16_sve_n_tile();
    const int n_tiles = N_r / n_tile;
    num_threads = bf16_effective_threads(num_threads, M, K_r, N_r, n_tiles);
    const size_t a_stride = bf16_a_reorder_stride(K_r);
    const int no_reorder = bf16_use_no_reorder_for_shape(M, K_r, n_tiles);
    const int use_n_split = bf16_use_n_split(M, K_r, N_r, n_tiles, num_threads);
    num_threads = bf16_clamp_threads_for_split(num_threads, M, n_tiles,
                                               use_n_split);
    const int m_blocks = (M + 7) / 8;

    if (!use_n_split) {
        if (!no_reorder) {
            bf16_t *scratch_pool = (bf16_t *)aligned_alloc_64(
                (size_t)num_threads * a_stride * sizeof(bf16_t));
            if (scratch_pool) {
                #pragma omp parallel num_threads(num_threads)
                {
                    int tid = omp_get_thread_num();
                    int nth = omp_get_num_threads();
                    int start = 0;
                    int end = 0;
                    bf16_t *scratch = scratch_pool + (size_t)tid * a_stride;
                    bf16_static_range(m_blocks, tid, nth, &start, &end);
                    for (int b = start; b < end; b++) {
                        int m0 = b * 8;
                        int mb = M - m0 < 8 ? M - m0 : 8;
                        bf16_pack_A_sve_block(A + (size_t)m0 * K_r,
                                              scratch, mb, K_r);
                        bf16_dispatch_f32(A + (size_t)m0 * K_r, B_reo,
                                          C + (size_t)m0 * N_r, mb, K_r,
                                          N_r, N_r, 1, bias, scratch);
                    }
                }
                free(scratch_pool);
                return;
            }
        }

        bf16_t *A_reo_pool = no_reorder ? NULL :
            bf16_prepare_A_reorder_pool(A, M, K_r, num_threads);
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int m0 = 0; m0 < M; m0 += 8) {
            int mb = M - m0 < 8 ? M - m0 : 8;
            bf16_t *A_reo = A_reo_pool ?
                A_reo_pool + (size_t)(m0 / 8) * a_stride : NULL;
            bf16_dispatch_f32(A + (size_t)m0 * K_r, B_reo,
                              C + (size_t)m0 * N_r, mb, K_r, N_r, N_r,
                              1, bias, A_reo);
        }
        free(A_reo_pool);
    } else {
        bf16_t *A_reo_pool = no_reorder ? NULL :
            (bf16_t *)aligned_alloc_64((size_t)m_blocks * a_stride * sizeof(bf16_t));
        #pragma omp parallel num_threads(num_threads)
        {
            if (A_reo_pool) {
                #pragma omp for schedule(static)
                for (int b = 0; b < m_blocks; b++) {
                    int m0 = b * 8;
                    int mb = M - m0 < 8 ? M - m0 : 8;
                    bf16_pack_A_sve_block(A + (size_t)m0 * K_r,
                                          A_reo_pool + (size_t)b * a_stride,
                                          mb, K_r);
                }
            }

            #pragma omp for schedule(static)
            for (int t = 0; t < n_tiles; t++) {
                int n0 = t * n_tile;
                for (int m0 = 0; m0 < M; m0 += 8) {
                    int mb = M - m0 < 8 ? M - m0 : 8;
                    bf16_t *A_reo = A_reo_pool ?
                        A_reo_pool + (size_t)(m0 / 8) * a_stride : NULL;
                    bf16_dispatch_f32(A + (size_t)m0 * K_r,
                                      B_reo + (size_t)t * K_r * n_tile,
                                      C + (size_t)m0 * N_r + n0,
                                      mb, K_r, n_tile, N_r, 1, bias + n0, A_reo);
                }
            }
        }
        free(A_reo_pool);
    }
}

void bf16gemm_mt_dispatch_nld_b(const bf16_t *A, const bf16_t *B_reo,
                                bf16_t *C, int M, int K_r, int N_r,
                                int num_threads) {
    const int n_tile = bf16_sve_n_tile();
    const int n_tiles = N_r / n_tile;
    num_threads = bf16_effective_threads(num_threads, M, K_r, N_r, n_tiles);
    const size_t a_stride = bf16_a_reorder_stride(K_r);
    const int no_reorder = bf16_use_no_reorder_for_shape(M, K_r, n_tiles);
    const int use_n_split = bf16_use_n_split(M, K_r, N_r, n_tiles, num_threads);
    num_threads = bf16_clamp_threads_for_split(num_threads, M, n_tiles,
                                               use_n_split);
    const int m_blocks = (M + 7) / 8;

    if (!use_n_split) {
        if (!no_reorder) {
            bf16_t *scratch_pool = (bf16_t *)aligned_alloc_64(
                (size_t)num_threads * a_stride * sizeof(bf16_t));
            if (scratch_pool) {
                #pragma omp parallel num_threads(num_threads)
                {
                    int tid = omp_get_thread_num();
                    int nth = omp_get_num_threads();
                    int start = 0;
                    int end = 0;
                    bf16_t *scratch = scratch_pool + (size_t)tid * a_stride;
                    bf16_static_range(m_blocks, tid, nth, &start, &end);
                    for (int b = start; b < end; b++) {
                        int m0 = b * 8;
                        int mb = M - m0 < 8 ? M - m0 : 8;
                        bf16_pack_A_sve_block(A + (size_t)m0 * K_r,
                                              scratch, mb, K_r);
                        bf16_dispatch_bf16(A + (size_t)m0 * K_r, B_reo,
                                           C + (size_t)m0 * N_r, mb, K_r,
                                           N_r, N_r, 1, scratch);
                    }
                }
                free(scratch_pool);
                return;
            }
        }

        bf16_t *A_reo_pool = no_reorder ? NULL :
            bf16_prepare_A_reorder_pool(A, M, K_r, num_threads);
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int m0 = 0; m0 < M; m0 += 8) {
            int mb = M - m0 < 8 ? M - m0 : 8;
            bf16_t *A_reo = A_reo_pool ?
                A_reo_pool + (size_t)(m0 / 8) * a_stride : NULL;
            bf16_dispatch_bf16(A + (size_t)m0 * K_r, B_reo,
                               C + (size_t)m0 * N_r, mb, K_r, N_r, N_r,
                               1, A_reo);
        }
        free(A_reo_pool);
    } else {
        bf16_t *A_reo_pool = no_reorder ? NULL :
            (bf16_t *)aligned_alloc_64((size_t)m_blocks * a_stride * sizeof(bf16_t));
        #pragma omp parallel num_threads(num_threads)
        {
            if (A_reo_pool) {
                #pragma omp for schedule(static)
                for (int b = 0; b < m_blocks; b++) {
                    int m0 = b * 8;
                    int mb = M - m0 < 8 ? M - m0 : 8;
                    bf16_pack_A_sve_block(A + (size_t)m0 * K_r,
                                          A_reo_pool + (size_t)b * a_stride,
                                          mb, K_r);
                }
            }

            #pragma omp for schedule(static)
            for (int t = 0; t < n_tiles; t++) {
                int n0 = t * n_tile;
                for (int m0 = 0; m0 < M; m0 += 8) {
                    int mb = M - m0 < 8 ? M - m0 : 8;
                    bf16_t *A_reo = A_reo_pool ?
                        A_reo_pool + (size_t)(m0 / 8) * a_stride : NULL;
                    bf16_dispatch_bf16(A + (size_t)m0 * K_r,
                                       B_reo + (size_t)t * K_r * n_tile,
                                       C + (size_t)m0 * N_r + n0,
                                       mb, K_r, n_tile, N_r, 1, A_reo);
                }
            }
        }
        free(A_reo_pool);
    }
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

void bf16gemm_sve_pack_B(const bf16_t *B, bf16_t *B_reo, int K, int N) {
    bf16_pack_B(B, B_reo, K, N);
}

void bf16gemm_sve_dispatch_f32(const bf16_t *A, const bf16_t *B_reo,
                               f32_t *C, int M, int K_r, int N_r,
                               int nthreads) {
    bf16gemm_mt_dispatch(A, B_reo, C, M, K_r, N_r, nthreads);
}

void bf16gemm_sve_dispatch_bias_f32(const bf16_t *A, const bf16_t *B_reo,
                                    f32_t *C, int M, int K_r, int N_r,
                                    int nthreads, const f32_t *bias) {
    bf16gemm_mt_dispatch_bias_f(A, B_reo, C, M, K_r, N_r, nthreads, bias);
}

void bf16gemm_sve_dispatch_bf16(const bf16_t *A, const bf16_t *B_reo,
                                bf16_t *C, int M, int K_r, int N_r,
                                int nthreads) {
    bf16gemm_mt_dispatch_nld_b(A, B_reo, C, M, K_r, N_r, nthreads);
}

void bf16gemm_sve_f32(const bf16_t *A_orig, const bf16_t *B_orig,
                      f32_t *C, int M, int K, int N, int nthreads) {
    bf16gemm_mt(A_orig, B_orig, C, M, K, N, nthreads);
}

void bf16gemm_sve_bias_f32(const bf16_t *A_orig, const bf16_t *B_orig,
                           f32_t *C, int M, int K, int N, int nthreads,
                           const f32_t *bias) {
    bf16gemm_mt_bias_f(A_orig, B_orig, C, M, K, N, nthreads, bias);
}

void bf16gemm_sve_bf16(const bf16_t *A_orig, const bf16_t *B_orig,
                       bf16_t *C, int M, int K, int N, int nthreads) {
    bf16gemm_mt_nld_b(A_orig, B_orig, C, M, K, N, nthreads);
}
