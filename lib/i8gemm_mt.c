// i8gemm_mt.c — OpenMP Multi-threaded I8 GEMM (library)
//
// Auto-selects M-tiling or N-tiling based on shape.
// B must be pre-packed by caller (i8_pack_B) outside the timed region.
//
// Exported API:
//   void i8_pack_B(const i8_t *B, i8_t *B_reo, int K, int N);
//   void i8gemm_mt_dispatch(const i8_t *A, const i8_t *B_reo,
//                            i32_t *C, int M, int K_r, int N_r, int nthreads);
//   void i8gemm_mt(const i8_t *A, const i8_t *B,
//                   i32_t *C, int M, int K, int N, int nthreads);
//
// Build:
//   cc -c i8gemm_mt.c -march=armv8.6-a+i8mm -O2 -Wall -fopenmp
//   (link with i8gemm_k.S and -fopenmp -lm)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>

#include "gemm_params.h"

// ═══════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════
typedef int8_t  i8_t;
typedef int32_t i32_t;
typedef float    f32_t;

static size_t round_up_64(size_t size) {
    return (size + 63u) & ~(size_t)63u;
}

static void *aligned_alloc_64(size_t size) {
    if (size == 0)
        size = 64;
    return aligned_alloc(64, round_up_64(size));
}

static int i8_effective_threads(int num_threads, int M, int n_tiles) {
    if (num_threads <= 0) num_threads = omp_get_max_threads();
    if (num_threads <= 0) num_threads = 1;

    const char *clamp_env = getenv("I8_GEMM_CLAMP_THREADS");
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

static int i8_use_n_split(int M, int K_r, int N_r, int n_tiles,
                          int num_threads) {
    const char *split = getenv("I8_GEMM_SPLIT");
    if (split) {
        if (strcmp(split, "m") == 0)
            return 0;
        if (strcmp(split, "n") == 0)
            return n_tiles >= num_threads;
    }

    if (num_threads <= 1 || n_tiles < num_threads)
        return 0;

    /*
     * 80c opt-parts data shows large NEON I8 shapes with N ~= 2*M are much
     * faster with M splitting: N-split repeats the M sweep for every N shard
     * and exposes the online A-pack/store overhead.  Very wide N (for example
     * N >= 4*M) still prefers N splitting.
     */
    if (num_threads >= 16 && M >= 512 && K_r >= 512 && N_r <= M * 2)
        return 0;

    if (M / 8 < num_threads)
        return 1;

    /*
     * ACL-style probing shows that medium-M, wide-N NEON I8 shapes should not
     * always move to pure N-split just because the packed-B panel is large.
     * With 4/8 cores, M-split keeps each worker on a long contiguous N sweep
     * and avoids repeating the M sweep per N shard.  This fixes shapes such as
     * 128x512x1024 and 256x512x1024, where old N-split under-scaled badly.
     */
    if (num_threads >= 4 && M >= 128 && M <= 256 &&
        K_r >= 512 && N_r <= M * 8)
        return 0;

    const size_t b_panel_bytes = (size_t)K_r * (size_t)N_r * sizeof(i8_t);
    return b_panel_bytes >= 512u * 1024u && N_r >= M * 2;
}

static int i8_requested_2d_split(void) {
    const char *split = getenv("I8_GEMM_SPLIT");
    return split && strcmp(split, "2d") == 0;
}

static void i8_split_2d_threads(int max_threads, int m_work, int n_work,
                                int *m_threads, int *n_threads) {
    if (max_threads < 1)
        max_threads = 1;
    if (m_work < 1)
        m_work = 1;
    if (n_work < 1)
        n_work = 1;

    int best_m = 1;
    int best_n = max_threads;
    double best_score = 1.0e300;
    for (int mt = 1; mt <= max_threads; mt++) {
        if (max_threads % mt != 0)
            continue;
        int nt = max_threads / mt;
        if (mt > m_work || nt > n_work)
            continue;
        double m_per = (double)m_work / (double)mt;
        double n_per = (double)n_work / (double)nt;
        double score = m_per > n_per ? m_per / n_per : n_per / m_per;
        if (score < best_score) {
            best_score = score;
            best_m = mt;
            best_n = nt;
        }
    }
    *m_threads = best_m;
    *n_threads = best_n;
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel declarations (from i8gemm_k.S)
// ═══════════════════════════════════════════════════════════════════════
void i8gemm_k_ld (const i8_t *A, const i8_t *B_reo,
                  i32_t *C, i8_t *A_reorder,
                  const gemm_params_t *params);
void i8gemm_k_ld1(const i8_t *A, const i8_t *B_reo,
                  i32_t *C, i8_t *A_reorder,
                  const gemm_params_t *params);
void i8gemm_k_ld2(const i8_t *A, const i8_t *B_reo,
                  i32_t *C, i8_t *A_reorder,
                  const gemm_params_t *params);
void i8gemm_k_ld4(const i8_t *A, const i8_t *B_reo,
                  i32_t *C, i8_t *A_reorder,
                  const gemm_params_t *params);
void i8gemm_k_zero (const i8_t *A, const i8_t *B_reo,
                    i32_t *C, i8_t *A_reorder,
                    const gemm_params_t *params);
void i8gemm_k_zero1(const i8_t *A, const i8_t *B_reo,
                    i32_t *C, i8_t *A_reorder,
                    const gemm_params_t *params);
void i8gemm_k_zero2(const i8_t *A, const i8_t *B_reo,
                    i32_t *C, i8_t *A_reorder,
                    const gemm_params_t *params);
void i8gemm_k_zero4(const i8_t *A, const i8_t *B_reo,
                    i32_t *C, i8_t *A_reorder,
                    const gemm_params_t *params);
void i8gemm_k_reo_ld (const i8_t *A, const i8_t *B_reo,
                      i32_t *C, i8_t *A_reorder,
                      const gemm_params_t *params);
void i8gemm_k_reo_ld1(const i8_t *A, const i8_t *B_reo,
                      i32_t *C, i8_t *A_reorder,
                      const gemm_params_t *params);
void i8gemm_k_reo_ld2(const i8_t *A, const i8_t *B_reo,
                      i32_t *C, i8_t *A_reorder,
                      const gemm_params_t *params);
void i8gemm_k_reo_ld4(const i8_t *A, const i8_t *B_reo,
                      i32_t *C, i8_t *A_reorder,
                      const gemm_params_t *params);
void i8gemm_k_reo_zero (const i8_t *A, const i8_t *B_reo,
                        i32_t *C, i8_t *A_reorder,
                        const gemm_params_t *params);
void i8gemm_k_reo_zero1(const i8_t *A, const i8_t *B_reo,
                        i32_t *C, i8_t *A_reorder,
                        const gemm_params_t *params);
void i8gemm_k_reo_zero2(const i8_t *A, const i8_t *B_reo,
                        i32_t *C, i8_t *A_reorder,
                        const gemm_params_t *params);
void i8gemm_k_reo_zero4(const i8_t *A, const i8_t *B_reo,
                        i32_t *C, i8_t *A_reorder,
                        const gemm_params_t *params);

void i8_pack_A_neon_m8_asm(const i8_t *A, i8_t *P, int K_r, int lda);
void i8_pack_A_neon_m4_asm(const i8_t *A, i8_t *P, int K_r, int lda);
void i8_pack_A_neon_m2_asm(const i8_t *A, i8_t *P, int K_r, int lda);
void i8_pack_A_neon_m1_asm(const i8_t *A, i8_t *P, int K_r, int lda);

static size_t i8_neon_a_reorder_bytes(int M, int K_r) {
    return (size_t)(M + 8) * (size_t)K_r * sizeof(i8_t);
}

static int i8_neon_pack_a_thread_count(int num_threads, int M, int K_r) {
    if (num_threads <= 1)
        return 1;

    int blocks = (M + 7) / 8;
    if (blocks < 1)
        blocks = 1;
    if (num_threads > blocks)
        num_threads = blocks;

    const char *env = getenv("I8_NEON_PACK_A_MIN_BYTES_PER_THREAD");
    size_t min_bytes = env ? (size_t)strtoull(env, NULL, 0) : 32u * 1024u;
    size_t bytes = (size_t)M * (size_t)K_r;
    while (num_threads > 1 &&
           bytes / (size_t)num_threads < min_bytes)
        num_threads--;

    return num_threads < 1 ? 1 : num_threads;
}

static void i8_pack_A_neon(const i8_t *A, i8_t *P, int M, int K_r,
                           int lda, int num_threads) {
    int m_full = (M / 8) * 8;
    int pack_threads = i8_neon_pack_a_thread_count(num_threads, M, K_r);

    #pragma omp parallel for num_threads(pack_threads) schedule(static)
    for (int m0 = 0; m0 < m_full; m0 += 8) {
        i8_pack_A_neon_m8_asm(A + (size_t)m0 * (size_t)lda,
                              P + (size_t)m0 * (size_t)K_r, K_r, lda);
    }

    int processed = m_full;
    int m_rem = M - processed;
    i8_t *tail = P + (size_t)processed * (size_t)K_r;
    const i8_t *At = A + (size_t)processed * (size_t)lda;

    if (m_rem >= 4) {
        i8_pack_A_neon_m4_asm(At, tail, K_r, lda);
        processed += 4;
        m_rem -= 4;
        At = A + (size_t)processed * (size_t)lda;
        tail = P + (size_t)processed * (size_t)K_r;
    }
    if (m_rem >= 2) {
        i8_pack_A_neon_m2_asm(At, tail, K_r, lda);
        processed += 2;
        m_rem -= 2;
        At = A + (size_t)processed * (size_t)lda;
        tail = P + (size_t)processed * (size_t)K_r;
    }
    if (m_rem >= 1)
        i8_pack_A_neon_m1_asm(At, tail, K_r, lda);
}

static int i8_neon_offline_a_reorder_mode(void) {
    const char *env = getenv("I8_NEON_OFFLINE_A_REORDER");
    if (!env)
        return 1;  // default: offline-pack A only for N-split.
    int mode = atoi(env);
    if (mode < 0)
        mode = 0;
    return mode;
}

static int i8_accumulate_c_mode(void) {
    const char *env = getenv("I8_GEMM_ACCUMULATE_C");
    return env ? atoi(env) != 0 : 0;
}

// ═══════════════════════════════════════════════════════════════════════
// i8_pack_B — Pack B matrix for smmla kernel (caller allocates B_reo)
//
// B_reo size: K * N int8 elements.
// Layout (outer→inner):
//   for each N-block (8 cols):
//     for each K-block (8 rows):
//       for each column j in N-block:
//         for each row i in K-block:
//           B[Kblock*8 + i][Nblock*8 + j]
//
// One N-block = (K/8) * 8 * 8 = K * 8 int8 values = K * 8 bytes.
// ═══════════════════════════════════════════════════════════════════════
void i8_pack_B(const i8_t *B, i8_t *B_reo, int K, int N) {
    int idx = 0;
    for (int cb = 0; cb < N / 8; ++cb)
        for (int rb = 0; rb < K / 8; ++rb)
            for (int j = 0; j < 8; ++j)
                for (int i = 0; i < 8; ++i)
                    B_reo[idx++] = B[(rb * 8 + i) * N + (cb * 8 + j)];
}

// ═══════════════════════════════════════════════════════════════════════
// i8_dispatch — Single-thread kernel dispatch (internal)
//
// Handles arbitrary M by decomposing into full-8 + tail-4/2/1 calls.
// ldc_global: C row stride in i32 elements (may differ from n for N-tiling).
// ═══════════════════════════════════════════════════════════════════════
static void i8_dispatch_core(const i8_t *A, const i8_t *B_reo,
                             i32_t *C, int m, int k, int n,
                             i8_t *A_reorder, int ldc_global,
                             int zero_c) {
    volatile gemm_params_t p;
    p.lda = k;  p.ldb = k;  p.ldc = ldc_global;
    int processed = 0;

    int m_full = (m / 8) * 8;
    if (m_full > 0) {
        p.m = m_full;  p.k = k;  p.n = n;
        if (zero_c)
            i8gemm_k_zero(A, B_reo, C, A_reorder,
                          (const gemm_params_t *)&p);
        else
            i8gemm_k_ld(A, B_reo, C, A_reorder,
                        (const gemm_params_t *)&p);
        processed = m_full;
    }

    int m_rem = m - processed;
    if (m_rem == 0) return;

    const i8_t *At = A + (uint64_t)processed * k;
    i32_t      *Ct = C + (uint64_t)processed * ldc_global;
    i8_t *A_reo_t  = A_reorder + (uint64_t)processed * k;

    if (m_rem >= 4) {
        p.m = 4;  p.k = k;  p.n = n;
        if (zero_c)
            i8gemm_k_zero4(At, B_reo, Ct, A_reo_t,
                           (const gemm_params_t *)&p);
        else
            i8gemm_k_ld4(At, B_reo, Ct, A_reo_t,
                         (const gemm_params_t *)&p);
        processed += 4; m_rem -= 4;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * ldc_global;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 2) {
        p.m = 2;  p.k = k;  p.n = n;
        if (zero_c)
            i8gemm_k_zero2(At, B_reo, Ct, A_reo_t,
                           (const gemm_params_t *)&p);
        else
            i8gemm_k_ld2(At, B_reo, Ct, A_reo_t,
                         (const gemm_params_t *)&p);
        processed += 2; m_rem -= 2;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * ldc_global;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 1) {
        p.m = 1;  p.k = k;  p.n = n;
        if (zero_c)
            i8gemm_k_zero1(At, B_reo, Ct, A_reo_t,
                           (const gemm_params_t *)&p);
        else
            i8gemm_k_ld1(At, B_reo, Ct, A_reo_t,
                         (const gemm_params_t *)&p);
    }
}

static void i8_dispatch_reo_core(const i8_t *A, const i8_t *B_reo,
                                 i32_t *C, int m, int k, int n,
                                 i8_t *A_reorder, int ldc_global,
                                 int zero_c) {
    volatile gemm_params_t p;
    p.lda = k;  p.ldb = k;  p.ldc = ldc_global;
    int processed = 0;

    int m_full = (m / 8) * 8;
    if (m_full > 0) {
        p.m = m_full;  p.k = k;  p.n = n;
        if (zero_c)
            i8gemm_k_reo_zero(A, B_reo, C, A_reorder,
                              (const gemm_params_t *)&p);
        else
            i8gemm_k_reo_ld(A, B_reo, C, A_reorder,
                            (const gemm_params_t *)&p);
        processed = m_full;
    }

    int m_rem = m - processed;
    if (m_rem == 0) return;

    const i8_t *At = A + (uint64_t)processed * k;
    i32_t      *Ct = C + (uint64_t)processed * ldc_global;
    i8_t *A_reo_t  = A_reorder + (uint64_t)processed * k;

    if (m_rem >= 4) {
        p.m = 4;  p.k = k;  p.n = n;
        if (zero_c)
            i8gemm_k_reo_zero4(At, B_reo, Ct, A_reo_t,
                               (const gemm_params_t *)&p);
        else
            i8gemm_k_reo_ld4(At, B_reo, Ct, A_reo_t,
                             (const gemm_params_t *)&p);
        processed += 4; m_rem -= 4;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * ldc_global;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 2) {
        p.m = 2;  p.k = k;  p.n = n;
        if (zero_c)
            i8gemm_k_reo_zero2(At, B_reo, Ct, A_reo_t,
                               (const gemm_params_t *)&p);
        else
            i8gemm_k_reo_ld2(At, B_reo, Ct, A_reo_t,
                             (const gemm_params_t *)&p);
        processed += 2; m_rem -= 2;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * ldc_global;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 1) {
        p.m = 1;  p.k = k;  p.n = n;
        if (zero_c)
            i8gemm_k_reo_zero1(At, B_reo, Ct, A_reo_t,
                               (const gemm_params_t *)&p);
        else
            i8gemm_k_reo_ld1(At, B_reo, Ct, A_reo_t,
                             (const gemm_params_t *)&p);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// tile_M_i8 — M-tiling: split M rows across threads
// ═══════════════════════════════════════════════════════════════════════
static void tile_M_i8(const i8_t *A, const i8_t *B_reo,
                       i32_t *C, int M, int K_r, int N_r,
                       i8_t *A_reo_pool, int num_threads, int prepacked_a,
                       int zero_c) {
    int M8 = M / 8;
    int M_rem = M - M8 * 8;
    int blocks_per_thread = M8 / num_threads;
    int extra_blocks       = M8 % num_threads;

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        int start_block, my_blocks;
        if (tid < extra_blocks) {
            start_block = tid * (blocks_per_thread + 1);
            my_blocks   = blocks_per_thread + 1;
        } else {
            start_block = extra_blocks * (blocks_per_thread + 1)
                        + (tid - extra_blocks) * blocks_per_thread;
            my_blocks   = blocks_per_thread;
        }

        int my_m = my_blocks * 8 + ((tid == num_threads - 1) ? M_rem : 0);

        if (my_m > 0) {
            int my_m_start = start_block * 8;
            const i8_t *my_A = A + (uint64_t)my_m_start * K_r;
            i32_t      *my_C = C + (uint64_t)my_m_start * N_r;
            i8_t *my_A_reo  = A_reo_pool + (uint64_t)my_m_start * K_r;

            if (prepacked_a)
                i8_dispatch_reo_core(my_A, B_reo, my_C, my_m, K_r, N_r,
                                     my_A_reo, /*ldc_global=*/N_r,
                                     zero_c);
            else
                i8_dispatch_core(my_A, B_reo, my_C, my_m, K_r, N_r,
                                 my_A_reo, /*ldc_global=*/N_r, zero_c);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// tile_N_i8 — N-tiling: split N columns across threads (for small M)
// ═══════════════════════════════════════════════════════════════════════
static void tile_N_i8(const i8_t *A, const i8_t *B_reo,
                       i32_t *C, int M, int K_r, int N_r,
                       i8_t *A_reo_pool, int num_threads, int prepacked_a,
                       int zero_c) {
    int N8 = N_r / 8;
    int blocks_per_thread = N8 / num_threads;
    int extra_blocks       = N8 % num_threads;

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        int start_block, my_blocks;
        if (tid < extra_blocks) {
            start_block = tid * (blocks_per_thread + 1);
            my_blocks   = blocks_per_thread + 1;
        } else {
            start_block = extra_blocks * (blocks_per_thread + 1)
                        + (tid - extra_blocks) * blocks_per_thread;
            my_blocks   = blocks_per_thread;
        }

        if (my_blocks > 0) {
            int my_n       = my_blocks * 8;
            int my_n_start = start_block * 8;

            i8_t *my_A_reo = prepacked_a ? A_reo_pool :
                A_reo_pool + (size_t)tid * (size_t)(M + 8) * (size_t)K_r;
            if (my_A_reo) {
                // B: each N-block = K_r * 8 int8 values (= K_r * 8 bytes)
                const i8_t *my_B = B_reo + (uint64_t)start_block * K_r * 8;
                // C: column offset; ldc_global = N_r (full row stride)
                i32_t *my_C = C + my_n_start;

                if (prepacked_a)
                    i8_dispatch_reo_core(A, my_B, my_C, M, K_r, my_n,
                                         my_A_reo, /*ldc_global=*/N_r,
                                         zero_c);
                else
                    i8_dispatch_core(A, my_B, my_C, M, K_r, my_n,
                                     my_A_reo, /*ldc_global=*/N_r, zero_c);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// tile_2D_i8 — ACL-style 2D split across M blocks and N panels.
// Requires prepacked A so every M/N task can reuse the same A panel.
// ═══════════════════════════════════════════════════════════════════════
static void tile_2D_i8(const i8_t *A, const i8_t *B_reo,
                       i32_t *C, int M, int K_r, int N_r,
                       i8_t *A_reo_pool, int num_threads,
                       int m_threads, int n_threads, int zero_c) {
    int M8 = M / 8;
    int M_rem = M - M8 * 8;
    int N8 = N_r / 8;

    #pragma omp parallel for collapse(2) num_threads(num_threads) schedule(static)
    for (int ni = 0; ni < n_threads; ni++) {
        for (int mi = 0; mi < m_threads; mi++) {
            int m_block_start = (M8 * mi) / m_threads;
            int m_block_end = (M8 * (mi + 1)) / m_threads;
            int n_block_start = (N8 * ni) / n_threads;
            int n_block_end = (N8 * (ni + 1)) / n_threads;

            int my_m = (m_block_end - m_block_start) * 8;
            if (mi == m_threads - 1)
                my_m += M_rem;
            int my_n = (n_block_end - n_block_start) * 8;
            if (my_m <= 0 || my_n <= 0)
                continue;

            int m_start = m_block_start * 8;
            int n_start = n_block_start * 8;
            const i8_t *my_A = A + (uint64_t)m_start * K_r;
            const i8_t *my_B = B_reo + (uint64_t)n_block_start * K_r * 8;
            i32_t *my_C = C + (uint64_t)m_start * N_r + n_start;
            i8_t *my_A_reo = A_reo_pool + (uint64_t)m_start * K_r;

            i8_dispatch_reo_core(my_A, my_B, my_C, my_m, K_r, my_n,
                                 my_A_reo, /*ldc_global=*/N_r, zero_c);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// i8gemm_mt_dispatch — Core multi-threaded compute (no allocation, no pack)
//
// A:   M × K_r  row-major int8 (padded to K_r multiple of 16)
// B_reo: pre-packed by i8_pack_B, K_r × N_r int8
// C:   M × N_r  row-major int32 (output, zero-init before call)
// K_r: must be multiple of 16 (smmla requirement)
// N_r: must be multiple of 8
//
// Place this inside the timed benchmark region.
// ═══════════════════════════════════════════════════════════════════════
void i8gemm_mt_dispatch(const i8_t *A, const i8_t *B_reo,
                         i32_t *C, int M, int K_r, int N_r,
                         int num_threads) {
    const int n_tiles = N_r / 8;
    num_threads = i8_effective_threads(num_threads, M, n_tiles);
    const int use_n_split = i8_use_n_split(M, K_r, N_r, n_tiles, num_threads);
    const int use_2d_split = i8_requested_2d_split() &&
        num_threads > 1 && M >= 16 && N_r >= 16;
    const int offline_mode = i8_neon_offline_a_reorder_mode();
    const int prepack_a = use_2d_split || offline_mode > 1 ||
        (offline_mode == 1 && use_n_split);
    const int zero_c = !i8_accumulate_c_mode();

    if (use_2d_split) {
        int m_threads = 1;
        int n_threads = num_threads;
        i8_split_2d_threads(num_threads, (M + 7) / 8, n_tiles,
                            &m_threads, &n_threads);
        i8_t *A_reo_pool = (i8_t *)aligned_alloc_64(
            i8_neon_a_reorder_bytes(M, K_r));
        if (!A_reo_pool) return;
        i8_pack_A_neon(A, A_reo_pool, M, K_r, K_r, num_threads);
        tile_2D_i8(A, B_reo, C, M, K_r, N_r, A_reo_pool, num_threads,
                   m_threads, n_threads, zero_c);
        free(A_reo_pool);
    } else if (!use_n_split) {
        // M-tiling: one shared A_reorder pool, partitioned by M rows.
        // ld1 tail kernel writes zero-padded q-regs (2× expansion per row);
        // use 2× allocation to avoid overflow for non-multiple-of-8 M.
        i8_t *A_reo_pool = (i8_t *)aligned_alloc_64(
            i8_neon_a_reorder_bytes(M, K_r));
        if (!A_reo_pool) return;
        if (prepack_a)
            i8_pack_A_neon(A, A_reo_pool, M, K_r, K_r, num_threads);
        tile_M_i8(A, B_reo, C, M, K_r, N_r, A_reo_pool, num_threads,
                  prepack_a, zero_c);
        free(A_reo_pool);
    } else {
        size_t copies = prepack_a ? 1u : (size_t)num_threads;
        i8_t *A_reo_pool = (i8_t *)aligned_alloc_64(
            copies * i8_neon_a_reorder_bytes(M, K_r));
        if (!A_reo_pool) return;
        if (prepack_a)
            i8_pack_A_neon(A, A_reo_pool, M, K_r, K_r, num_threads);
        tile_N_i8(A, B_reo, C, M, K_r, N_r, A_reo_pool, num_threads,
                  prepack_a, zero_c);
        free(A_reo_pool);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// i8gemm_mt — Convenience wrapper: pad, pack B, then dispatch
//
// A_orig: M×K row-major int8 (K need not be multiple of 16)
// B_orig: K×N row-major int8 (K need not be multiple of 16)
// C:      M×N row-major int32 output (zero-init before call)
//
// Packs B internally — for repeated calls with the same B, prefer
// i8_pack_B + i8gemm_mt_dispatch to avoid re-packing.
// ═══════════════════════════════════════════════════════════════════════
void i8gemm_mt(const i8_t *A_orig, const i8_t *B_orig,
                i32_t *C, int M, int K, int N,
                int num_threads) {
    int K_r = ((K + 15) / 16) * 16;  if (K_r < 16) K_r = 16;
    int N_r = ((N + 7)  / 8)  * 8;   if (N_r < 8)   N_r = 8;

    // Pack B
    i8_t *B_reo = (i8_t *)aligned_alloc_64(
        (size_t)K_r * N_r * sizeof(i8_t));
    if (!B_reo) return;

    i8_t *B_pad = NULL;
    const i8_t *B_use;
    if (K_r == K && N_r == N) {
        B_use = B_orig;
    } else {
        B_pad = (i8_t *)calloc((size_t)K_r * N_r, sizeof(i8_t));
        if (!B_pad) { free(B_reo); return; }
        for (int i = 0; i < K; i++)
            memcpy(B_pad + i * N_r, B_orig + i * N, N * sizeof(i8_t));
        B_use = B_pad;
    }
    i8_pack_B(B_use, B_reo, K_r, N_r);
    free(B_pad);

    // Pad A if needed
    i8_t *A_pad = NULL;
    const i8_t *A_use;
    if (K_r == K) {
        A_use = A_orig;
    } else {
        A_pad = (i8_t *)calloc((size_t)M * K_r, sizeof(i8_t));
        if (!A_pad) { free(B_reo); return; }
        for (int i = 0; i < M; i++)
            memcpy(A_pad + i * K_r, A_orig + i * K, K * sizeof(i8_t));
        A_use = A_pad;
    }

    if (N_r == N) {
        i8gemm_mt_dispatch(A_use, B_reo, C, M, K_r, N_r, num_threads);
    } else {
        i32_t *C_pad = (i32_t *)calloc((size_t)M * N_r, sizeof(i32_t));
        if (!C_pad) { free(A_pad); free(B_reo); return; }
        i8gemm_mt_dispatch(A_use, B_reo, C_pad, M, K_r, N_r, num_threads);
        for (int i = 0; i < M; i++)
            memcpy(C + (size_t)i * N, C_pad + (size_t)i * N_r,
                   (size_t)N * sizeof(i32_t));
        free(C_pad);
    }

    free(A_pad);
    free(B_reo);
}

// ═══════════════════════════════════════════════════════════════════════
// FP32-output I8 GEMM (_f suffix) — same int8 A/B, float32 C output
// The kernel accumulator is int32 (smmla), then converted to fp32 on store.
// B packing is identical to the i32 path (B is still int8).
// ═══════════════════════════════════════════════════════════════════════

// ── Kernel declarations (from i8gemm_k.S) ──────────────────────────
void i8gemm_k_ld_f (const i8_t *A, const i8_t *B_reo,
                    f32_t *C, i8_t *A_reorder,
                    const gemm_params_t *params);
void i8gemm_k_ld1_f(const i8_t *A, const i8_t *B_reo,
                    f32_t *C, i8_t *A_reorder,
                  const gemm_params_t *params);
void i8gemm_k_ld2_f(const i8_t *A, const i8_t *B_reo,
                    f32_t *C, i8_t *A_reorder,
                  const gemm_params_t *params);
void i8gemm_k_ld4_f(const i8_t *A, const i8_t *B_reo,
                    f32_t *C, i8_t *A_reorder,
                  const gemm_params_t *params);
void i8gemm_k_reo_ld_f (const i8_t *A, const i8_t *B_reo,
                        f32_t *C, i8_t *A_reorder,
                        const gemm_params_t *params);
void i8gemm_k_reo_ld1_f(const i8_t *A, const i8_t *B_reo,
                        f32_t *C, i8_t *A_reorder,
                        const gemm_params_t *params);
void i8gemm_k_reo_ld2_f(const i8_t *A, const i8_t *B_reo,
                        f32_t *C, i8_t *A_reorder,
                        const gemm_params_t *params);
void i8gemm_k_reo_ld4_f(const i8_t *A, const i8_t *B_reo,
                        f32_t *C, i8_t *A_reorder,
                        const gemm_params_t *params);

// ── i8_dispatch_f — Single-thread fp32 kernel dispatch ─────────────
static void i8_dispatch_f(const i8_t *A, const i8_t *B_reo,
                           f32_t *C, int m, int k, int n,
                           i8_t *A_reorder, int ldc_global) {
    volatile gemm_params_t p;
    p.lda = k;  p.ldb = k;  p.ldc = ldc_global;
    int processed = 0;

    int m_full = (m / 8) * 8;
    if (m_full > 0) {
        p.m = m_full;  p.k = k;  p.n = n;
        i8gemm_k_ld_f(A, B_reo, C, A_reorder, (const gemm_params_t *)&p);
        processed = m_full;
    }

    int m_rem = m - processed;
    if (m_rem == 0) return;

    const i8_t *At = A + (uint64_t)processed * k;
    f32_t      *Ct = C + (uint64_t)processed * ldc_global;
    i8_t *A_reo_t  = A_reorder + (uint64_t)processed * k;

    if (m_rem >= 4) {
        p.m = 4;  p.k = k;  p.n = n;
        i8gemm_k_ld4_f(At, B_reo, Ct, A_reo_t, (const gemm_params_t *)&p);
        processed += 4; m_rem -= 4;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * ldc_global;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 2) {
        p.m = 2;  p.k = k;  p.n = n;
        i8gemm_k_ld2_f(At, B_reo, Ct, A_reo_t, (const gemm_params_t *)&p);
        processed += 2; m_rem -= 2;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * ldc_global;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 1) {
        p.m = 1;  p.k = k;  p.n = n;
        i8gemm_k_ld1_f(At, B_reo, Ct, A_reo_t, (const gemm_params_t *)&p);
    }
}

static void i8_dispatch_reo_f(const i8_t *A, const i8_t *B_reo,
                              f32_t *C, int m, int k, int n,
                              i8_t *A_reorder, int ldc_global) {
    volatile gemm_params_t p;
    p.lda = k;  p.ldb = k;  p.ldc = ldc_global;
    int processed = 0;

    int m_full = (m / 8) * 8;
    if (m_full > 0) {
        p.m = m_full;  p.k = k;  p.n = n;
        i8gemm_k_reo_ld_f(A, B_reo, C, A_reorder, (const gemm_params_t *)&p);
        processed = m_full;
    }

    int m_rem = m - processed;
    if (m_rem == 0) return;

    const i8_t *At = A + (uint64_t)processed * k;
    f32_t      *Ct = C + (uint64_t)processed * ldc_global;
    i8_t *A_reo_t  = A_reorder + (uint64_t)processed * k;

    if (m_rem >= 4) {
        p.m = 4;  p.k = k;  p.n = n;
        i8gemm_k_reo_ld4_f(At, B_reo, Ct, A_reo_t, (const gemm_params_t *)&p);
        processed += 4; m_rem -= 4;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * ldc_global;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 2) {
        p.m = 2;  p.k = k;  p.n = n;
        i8gemm_k_reo_ld2_f(At, B_reo, Ct, A_reo_t, (const gemm_params_t *)&p);
        processed += 2; m_rem -= 2;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * ldc_global;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 1) {
        p.m = 1;  p.k = k;  p.n = n;
        i8gemm_k_reo_ld1_f(At, B_reo, Ct, A_reo_t, (const gemm_params_t *)&p);
    }
}

// ── tile_M_i8_f — M-tiling for fp32 output ─────────────────────────
static void tile_M_i8_f(const i8_t *A, const i8_t *B_reo,
                         f32_t *C, int M, int K_r, int N_r,
                         i8_t *A_reo_pool, int num_threads, int prepacked_a) {
    int M8 = M / 8;
    int M_rem = M - M8 * 8;
    int blocks_per_thread = M8 / num_threads;
    int extra_blocks       = M8 % num_threads;

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        int start_block, my_blocks;
        if (tid < extra_blocks) {
            start_block = tid * (blocks_per_thread + 1);
            my_blocks   = blocks_per_thread + 1;
        } else {
            start_block = extra_blocks * (blocks_per_thread + 1)
                        + (tid - extra_blocks) * blocks_per_thread;
            my_blocks   = blocks_per_thread;
        }

        int my_m = my_blocks * 8 + ((tid == num_threads - 1) ? M_rem : 0);

        if (my_m > 0) {
            int my_m_start = start_block * 8;
            const i8_t *my_A = A + (uint64_t)my_m_start * K_r;
            f32_t      *my_C = C + (uint64_t)my_m_start * N_r;
            i8_t *my_A_reo  = A_reo_pool + (uint64_t)my_m_start * K_r;

            if (prepacked_a)
                i8_dispatch_reo_f(my_A, B_reo, my_C, my_m, K_r, N_r,
                                  my_A_reo, /*ldc_global=*/N_r);
            else
                i8_dispatch_f(my_A, B_reo, my_C, my_m, K_r, N_r,
                              my_A_reo, /*ldc_global=*/N_r);
        }
    }
}

// ── tile_N_i8_f — N-tiling for fp32 output ─────────────────────────
static void tile_N_i8_f(const i8_t *A, const i8_t *B_reo,
                         f32_t *C, int M, int K_r, int N_r,
                         i8_t *A_reo_pool, int num_threads, int prepacked_a) {
    int N8 = N_r / 8;
    int blocks_per_thread = N8 / num_threads;
    int extra_blocks       = N8 % num_threads;

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        int start_block, my_blocks;
        if (tid < extra_blocks) {
            start_block = tid * (blocks_per_thread + 1);
            my_blocks   = blocks_per_thread + 1;
        } else {
            start_block = extra_blocks * (blocks_per_thread + 1)
                        + (tid - extra_blocks) * blocks_per_thread;
            my_blocks   = blocks_per_thread;
        }

        if (my_blocks > 0) {
            int my_n       = my_blocks * 8;
            int my_n_start = start_block * 8;

            i8_t *my_A_reo = prepacked_a ? A_reo_pool :
                A_reo_pool + (size_t)tid * (size_t)(M + 8) * (size_t)K_r;
            if (my_A_reo) {
                const i8_t *my_B = B_reo + (uint64_t)start_block * K_r * 8;
                f32_t *my_C = C + my_n_start;

                if (prepacked_a)
                    i8_dispatch_reo_f(A, my_B, my_C, M, K_r, my_n,
                                      my_A_reo, /*ldc_global=*/N_r);
                else
                    i8_dispatch_f(A, my_B, my_C, M, K_r, my_n,
                                  my_A_reo, /*ldc_global=*/N_r);
            }
        }
    }
}

// ── i8gemm_mt_dispatch_f — Core multi-threaded fp32 compute ────────
void i8gemm_mt_dispatch_f(const i8_t *A, const i8_t *B_reo,
                           f32_t *C, int M, int K_r, int N_r,
                           int num_threads) {
    const int n_tiles = N_r / 8;
    num_threads = i8_effective_threads(num_threads, M, n_tiles);
    const int use_n_split = i8_use_n_split(M, K_r, N_r, n_tiles, num_threads);
    const int offline_mode = i8_neon_offline_a_reorder_mode();
    const int prepack_a = offline_mode > 1 || (offline_mode == 1 && use_n_split);

    if (!use_n_split) {
        i8_t *A_reo_pool = (i8_t *)aligned_alloc_64(
            i8_neon_a_reorder_bytes(M, K_r));
        if (!A_reo_pool) return;
        if (prepack_a)
            i8_pack_A_neon(A, A_reo_pool, M, K_r, K_r, num_threads);
        tile_M_i8_f(A, B_reo, C, M, K_r, N_r, A_reo_pool, num_threads,
                    prepack_a);
        free(A_reo_pool);
    } else {
        size_t copies = prepack_a ? 1u : (size_t)num_threads;
        i8_t *A_reo_pool = (i8_t *)aligned_alloc_64(
            copies * i8_neon_a_reorder_bytes(M, K_r));
        if (!A_reo_pool) return;
        if (prepack_a)
            i8_pack_A_neon(A, A_reo_pool, M, K_r, K_r, num_threads);
        tile_N_i8_f(A, B_reo, C, M, K_r, N_r, A_reo_pool, num_threads,
                    prepack_a);
        free(A_reo_pool);
    }
}

// ── i8gemm_mt_f — Convenience wrapper: pad, pack B, fp32 dispatch ──
void i8gemm_mt_f(const i8_t *A_orig, const i8_t *B_orig,
                  f32_t *C, int M, int K, int N,
                  int num_threads) {
    int K_r = ((K + 15) / 16) * 16;  if (K_r < 16) K_r = 16;
    int N_r = ((N + 7)  / 8)  * 8;   if (N_r < 8)   N_r = 8;

    // Pack B (same as i32 path)
    i8_t *B_reo = (i8_t *)aligned_alloc_64(
        (size_t)K_r * N_r * sizeof(i8_t));
    if (!B_reo) return;

    i8_t *B_pad = NULL;
    const i8_t *B_use;
    if (K_r == K && N_r == N) {
        B_use = B_orig;
    } else {
        B_pad = (i8_t *)calloc((size_t)K_r * N_r, sizeof(i8_t));
        if (!B_pad) { free(B_reo); return; }
        for (int i = 0; i < K; i++)
            memcpy(B_pad + i * N_r, B_orig + i * N, N * sizeof(i8_t));
        B_use = B_pad;
    }
    i8_pack_B(B_use, B_reo, K_r, N_r);
    free(B_pad);

    // Pad A if needed
    i8_t *A_pad = NULL;
    const i8_t *A_use;
    if (K_r == K) {
        A_use = A_orig;
    } else {
        A_pad = (i8_t *)calloc((size_t)M * K_r, sizeof(i8_t));
        if (!A_pad) { free(B_reo); return; }
        for (int i = 0; i < M; i++)
            memcpy(A_pad + i * K_r, A_orig + i * K, K * sizeof(i8_t));
        A_use = A_pad;
    }

    if (N_r == N) {
        i8gemm_mt_dispatch_f(A_use, B_reo, C, M, K_r, N_r, num_threads);
    } else {
        f32_t *C_pad = (f32_t *)calloc((size_t)M * N_r, sizeof(f32_t));
        if (!C_pad) { free(A_pad); free(B_reo); return; }
        i8gemm_mt_dispatch_f(A_use, B_reo, C_pad, M, K_r, N_r, num_threads);
        for (int i = 0; i < M; i++)
            memcpy(C + (size_t)i * N, C_pad + (size_t)i * N_r,
                   (size_t)N * sizeof(f32_t));
        free(C_pad);
    }

    free(A_pad);
    free(B_reo);
}

// ═══════════════════════════════════════════════════════════════════════
// I8 GEMM with fp32 bias (_bias_f suffix) — no C load, zero-init accums
//
// C[i][j] = sum_k(A[i][k] * B[k][j]) + bias[j]
//
// Accumulators start at zero (no C read). Bias is added in fp32 domain
// during STORE after int32→fp32 conversion, avoiding precision loss.
// A/B are int8 (same as other i8 variants). B packing is identical.
// ═══════════════════════════════════════════════════════════════════════

// ── Kernel declarations (from i8gemm_k_bias.S) ────────────────────────
void i8gemm_k_ld_bias_f (const i8_t *A, const i8_t *B_reo,
                          f32_t *C, i8_t *A_reorder,
                          const gemm_params_t *params,
                          const f32_t *bias);
void i8gemm_k_ld1_bias_f(const i8_t *A, const i8_t *B_reo,
                          f32_t *C, i8_t *A_reorder,
                          const gemm_params_t *params,
                          const f32_t *bias);
void i8gemm_k_ld2_bias_f(const i8_t *A, const i8_t *B_reo,
                          f32_t *C, i8_t *A_reorder,
                          const gemm_params_t *params,
                          const f32_t *bias);
void i8gemm_k_ld4_bias_f(const i8_t *A, const i8_t *B_reo,
                          f32_t *C, i8_t *A_reorder,
                          const gemm_params_t *params,
                          const f32_t *bias);
void i8gemm_k_reo_ld_bias_f (const i8_t *A, const i8_t *B_reo,
                              f32_t *C, i8_t *A_reorder,
                              const gemm_params_t *params,
                              const f32_t *bias);
void i8gemm_k_reo_ld1_bias_f(const i8_t *A, const i8_t *B_reo,
                              f32_t *C, i8_t *A_reorder,
                              const gemm_params_t *params,
                              const f32_t *bias);
void i8gemm_k_reo_ld2_bias_f(const i8_t *A, const i8_t *B_reo,
                              f32_t *C, i8_t *A_reorder,
                              const gemm_params_t *params,
                              const f32_t *bias);
void i8gemm_k_reo_ld4_bias_f(const i8_t *A, const i8_t *B_reo,
                              f32_t *C, i8_t *A_reorder,
                              const gemm_params_t *params,
                              const f32_t *bias);

// ── i8_dispatch_bias_f — Single-thread bias kernel dispatch ───────────
static void i8_dispatch_bias_f(const i8_t *A, const i8_t *B_reo,
                                f32_t *C, int m, int k, int n,
                                i8_t *A_reorder, int ldc_global,
                                const f32_t *bias) {
    volatile gemm_params_t p;
    p.lda = k;  p.ldb = k;  p.ldc = ldc_global;
    int processed = 0;

    int m_full = (m / 8) * 8;
    if (m_full > 0) {
        p.m = m_full;  p.k = k;  p.n = n;
        i8gemm_k_ld_bias_f(A, B_reo, C, A_reorder, (const gemm_params_t *)&p, bias);
        processed = m_full;
    }

    int m_rem = m - processed;
    if (m_rem == 0) return;

    const i8_t *At = A + (uint64_t)processed * k;
    f32_t      *Ct = C + (uint64_t)processed * ldc_global;
    i8_t *A_reo_t  = A_reorder + (uint64_t)processed * k;

    if (m_rem >= 4) {
        p.m = 4;  p.k = k;  p.n = n;
        i8gemm_k_ld4_bias_f(At, B_reo, Ct, A_reo_t, (const gemm_params_t *)&p, bias);
        processed += 4; m_rem -= 4;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * ldc_global;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 2) {
        p.m = 2;  p.k = k;  p.n = n;
        i8gemm_k_ld2_bias_f(At, B_reo, Ct, A_reo_t, (const gemm_params_t *)&p, bias);
        processed += 2; m_rem -= 2;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * ldc_global;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 1) {
        p.m = 1;  p.k = k;  p.n = n;
        i8gemm_k_ld1_bias_f(At, B_reo, Ct, A_reo_t, (const gemm_params_t *)&p, bias);
    }
}

static void i8_dispatch_reo_bias_f(const i8_t *A, const i8_t *B_reo,
                                   f32_t *C, int m, int k, int n,
                                   i8_t *A_reorder, int ldc_global,
                                   const f32_t *bias) {
    volatile gemm_params_t p;
    p.lda = k;  p.ldb = k;  p.ldc = ldc_global;
    int processed = 0;

    int m_full = (m / 8) * 8;
    if (m_full > 0) {
        p.m = m_full;  p.k = k;  p.n = n;
        i8gemm_k_reo_ld_bias_f(A, B_reo, C, A_reorder,
                               (const gemm_params_t *)&p, bias);
        processed = m_full;
    }

    int m_rem = m - processed;
    if (m_rem == 0) return;

    const i8_t *At = A + (uint64_t)processed * k;
    f32_t      *Ct = C + (uint64_t)processed * ldc_global;
    i8_t *A_reo_t  = A_reorder + (uint64_t)processed * k;

    if (m_rem >= 4) {
        p.m = 4;  p.k = k;  p.n = n;
        i8gemm_k_reo_ld4_bias_f(At, B_reo, Ct, A_reo_t,
                                (const gemm_params_t *)&p, bias);
        processed += 4; m_rem -= 4;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * ldc_global;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 2) {
        p.m = 2;  p.k = k;  p.n = n;
        i8gemm_k_reo_ld2_bias_f(At, B_reo, Ct, A_reo_t,
                                (const gemm_params_t *)&p, bias);
        processed += 2; m_rem -= 2;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * ldc_global;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 1) {
        p.m = 1;  p.k = k;  p.n = n;
        i8gemm_k_reo_ld1_bias_f(At, B_reo, Ct, A_reo_t,
                                (const gemm_params_t *)&p, bias);
    }
}

// ── tile_M_i8_bias_f — M-tiling for bias fp32 output ──────────────────
static void tile_M_i8_bias_f(const i8_t *A, const i8_t *B_reo,
                              f32_t *C, int M, int K_r, int N_r,
                              i8_t *A_reo_pool, int num_threads,
                              const f32_t *bias, int prepacked_a) {
    int M8 = M / 8;
    int M_rem = M - M8 * 8;
    int blocks_per_thread = M8 / num_threads;
    int extra_blocks       = M8 % num_threads;

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        int start_block, my_blocks;
        if (tid < extra_blocks) {
            start_block = tid * (blocks_per_thread + 1);
            my_blocks   = blocks_per_thread + 1;
        } else {
            start_block = extra_blocks * (blocks_per_thread + 1)
                        + (tid - extra_blocks) * blocks_per_thread;
            my_blocks   = blocks_per_thread;
        }

        int my_m = my_blocks * 8 + ((tid == num_threads - 1) ? M_rem : 0);

        if (my_m > 0) {
            int my_m_start = start_block * 8;
            const i8_t *my_A = A + (uint64_t)my_m_start * K_r;
            f32_t      *my_C = C + (uint64_t)my_m_start * N_r;
            i8_t *my_A_reo  = A_reo_pool + (uint64_t)my_m_start * K_r;

            if (prepacked_a)
                i8_dispatch_reo_bias_f(my_A, B_reo, my_C, my_m, K_r, N_r,
                                       my_A_reo, /*ldc_global=*/N_r, bias);
            else
                i8_dispatch_bias_f(my_A, B_reo, my_C, my_m, K_r, N_r,
                                   my_A_reo, /*ldc_global=*/N_r, bias);
        }
    }
}

// ── tile_N_i8_bias_f — N-tiling for bias fp32 output ──────────────────
static void tile_N_i8_bias_f(const i8_t *A, const i8_t *B_reo,
                              f32_t *C, int M, int K_r, int N_r,
                              i8_t *A_reo_pool, int num_threads,
                              const f32_t *bias, int prepacked_a) {
    int N8 = N_r / 8;
    int blocks_per_thread = N8 / num_threads;
    int extra_blocks       = N8 % num_threads;

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        int start_block, my_blocks;
        if (tid < extra_blocks) {
            start_block = tid * (blocks_per_thread + 1);
            my_blocks   = blocks_per_thread + 1;
        } else {
            start_block = extra_blocks * (blocks_per_thread + 1)
                        + (tid - extra_blocks) * blocks_per_thread;
            my_blocks   = blocks_per_thread;
        }

        if (my_blocks > 0) {
            int my_n       = my_blocks * 8;
            int my_n_start = start_block * 8;

            i8_t *my_A_reo = prepacked_a ? A_reo_pool :
                A_reo_pool + (size_t)tid * (size_t)(M + 8) * (size_t)K_r;
            if (my_A_reo) {
                const i8_t   *my_B   = B_reo + (uint64_t)start_block * K_r * 8;
                f32_t        *my_C   = C + my_n_start;
                const f32_t  *my_bias = bias + my_n_start;

                if (prepacked_a)
                    i8_dispatch_reo_bias_f(A, my_B, my_C, M, K_r, my_n,
                                           my_A_reo, /*ldc_global=*/N_r,
                                           my_bias);
                else
                    i8_dispatch_bias_f(A, my_B, my_C, M, K_r, my_n,
                                       my_A_reo, /*ldc_global=*/N_r, my_bias);
            }
        }
    }
}

// ── i8gemm_mt_dispatch_bias_f — Core multi-threaded bias compute ──────
void i8gemm_mt_dispatch_bias_f(const i8_t *A, const i8_t *B_reo,
                                f32_t *C, int M, int K_r, int N_r,
                                int num_threads, const f32_t *bias) {
    const int n_tiles = N_r / 8;
    num_threads = i8_effective_threads(num_threads, M, n_tiles);
    const int use_n_split = i8_use_n_split(M, K_r, N_r, n_tiles, num_threads);
    const int offline_mode = i8_neon_offline_a_reorder_mode();
    const int prepack_a = offline_mode > 1 || (offline_mode == 1 && use_n_split);

    if (!use_n_split) {
        i8_t *A_reo_pool = (i8_t *)aligned_alloc_64(
            i8_neon_a_reorder_bytes(M, K_r));
        if (!A_reo_pool) return;
        if (prepack_a)
            i8_pack_A_neon(A, A_reo_pool, M, K_r, K_r, num_threads);
        tile_M_i8_bias_f(A, B_reo, C, M, K_r, N_r, A_reo_pool,
                          num_threads, bias, prepack_a);
        free(A_reo_pool);
    } else {
        size_t copies = prepack_a ? 1u : (size_t)num_threads;
        i8_t *A_reo_pool = (i8_t *)aligned_alloc_64(
            copies * i8_neon_a_reorder_bytes(M, K_r));
        if (!A_reo_pool) return;
        if (prepack_a)
            i8_pack_A_neon(A, A_reo_pool, M, K_r, K_r, num_threads);
        tile_N_i8_bias_f(A, B_reo, C, M, K_r, N_r, A_reo_pool,
                         num_threads, bias, prepack_a);
        free(A_reo_pool);
    }
}

// ── i8gemm_mt_bias_f — Convenience wrapper: pad, pack B, bias dispatch ─
void i8gemm_mt_bias_f(const i8_t *A_orig, const i8_t *B_orig,
                       f32_t *C, int M, int K, int N,
                       int num_threads, const f32_t *bias) {
    int K_r = ((K + 15) / 16) * 16;  if (K_r < 16) K_r = 16;
    int N_r = ((N + 7)  / 8)  * 8;   if (N_r < 8)   N_r = 8;

    // Pack B (same as other i8 variants)
    i8_t *B_reo = (i8_t *)aligned_alloc_64(
        (size_t)K_r * N_r * sizeof(i8_t));
    if (!B_reo) return;

    i8_t *B_pad = NULL;
    const i8_t *B_use;
    if (K_r == K && N_r == N) {
        B_use = B_orig;
    } else {
        B_pad = (i8_t *)calloc((size_t)K_r * N_r, sizeof(i8_t));
        if (!B_pad) { free(B_reo); return; }
        for (int i = 0; i < K; i++)
            memcpy(B_pad + i * N_r, B_orig + i * N, N * sizeof(i8_t));
        B_use = B_pad;
    }
    i8_pack_B(B_use, B_reo, K_r, N_r);
    free(B_pad);

    // Pad bias if needed (zero-extend for padding columns)
    f32_t *bias_pad = NULL;
    const f32_t *bias_use;
    if (N_r == N) {
        bias_use = bias;
    } else {
        bias_pad = (f32_t *)calloc((size_t)N_r, sizeof(f32_t));
        if (!bias_pad) { free(B_reo); return; }
        memcpy(bias_pad, bias, N * sizeof(f32_t));
        bias_use = bias_pad;
    }

    // Pad A if needed
    i8_t *A_pad = NULL;
    const i8_t *A_use;
    if (K_r == K) {
        A_use = A_orig;
    } else {
        A_pad = (i8_t *)calloc((size_t)M * K_r, sizeof(i8_t));
        if (!A_pad) { free(bias_pad); free(B_reo); return; }
        for (int i = 0; i < M; i++)
            memcpy(A_pad + i * K_r, A_orig + i * K, K * sizeof(i8_t));
        A_use = A_pad;
    }

    if (N_r == N) {
        i8gemm_mt_dispatch_bias_f(A_use, B_reo, C, M, K_r, N_r,
                                  num_threads, bias_use);
    } else {
        f32_t *C_pad = (f32_t *)calloc((size_t)M * N_r, sizeof(f32_t));
        if (!C_pad) { free(A_pad); free(bias_pad); free(B_reo); return; }
        i8gemm_mt_dispatch_bias_f(A_use, B_reo, C_pad, M, K_r, N_r,
                                  num_threads, bias_use);
        for (int i = 0; i < M; i++)
            memcpy(C + (size_t)i * N, C_pad + (size_t)i * N_r,
                   (size_t)N * sizeof(f32_t));
        free(C_pad);
    }

    free(A_pad);
    free(bias_pad);
    free(B_reo);
}
