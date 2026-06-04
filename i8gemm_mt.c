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

// ═══════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════
typedef int8_t  i8_t;
typedef int32_t i32_t;
typedef float    f32_t;

// ═══════════════════════════════════════════════════════════════════════
// Kernel declarations (from i8gemm_k.S)
// ═══════════════════════════════════════════════════════════════════════
void i8gemm_k_ld (const i8_t *A, const i8_t *B_reo,
                  i32_t *C, int m, int k, int n,
                  i8_t *A_reorder, int lda, int ldb, int ldc);
void i8gemm_k_ld1(const i8_t *A, const i8_t *B_reo,
                  i32_t *C, int m, int k, int n,
                  i8_t *A_reorder, int lda, int ldb, int ldc);
void i8gemm_k_ld2(const i8_t *A, const i8_t *B_reo,
                  i32_t *C, int m, int k, int n,
                  i8_t *A_reorder, int lda, int ldb, int ldc);
void i8gemm_k_ld4(const i8_t *A, const i8_t *B_reo,
                  i32_t *C, int m, int k, int n,
                  i8_t *A_reorder, int lda, int ldb, int ldc);

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
static void i8_dispatch(const i8_t *A, const i8_t *B_reo,
                         i32_t *C, int m, int k, int n,
                         i8_t *A_reorder, int ldc_global) {
    int lda = k;  // byte stride (int8 = 1 byte/elem)
    int ldb = k;
    int processed = 0;

    int m_full = (m / 8) * 8;
    if (m_full > 0) {
        i8gemm_k_ld(A, B_reo, C, m_full, k, n,
                    A_reorder, lda, ldb, ldc_global);
        processed = m_full;
    }

    int m_rem = m - processed;
    if (m_rem == 0) return;

    const i8_t *At = A + (uint64_t)processed * k;
    i32_t      *Ct = C + (uint64_t)processed * ldc_global;
    i8_t *A_reo_t  = A_reorder + (uint64_t)processed * k;

    if (m_rem >= 4) {
        i8gemm_k_ld4(At, B_reo, Ct, 4, k, n, A_reo_t, lda, ldb, ldc_global);
        processed += 4; m_rem -= 4;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * ldc_global;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 2) {
        i8gemm_k_ld2(At, B_reo, Ct, 2, k, n, A_reo_t, lda, ldb, ldc_global);
        processed += 2; m_rem -= 2;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * ldc_global;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 1) {
        i8gemm_k_ld1(At, B_reo, Ct, 1, k, n, A_reo_t, lda, ldb, ldc_global);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// tile_M_i8 — M-tiling: split M rows across threads
// ═══════════════════════════════════════════════════════════════════════
static void tile_M_i8(const i8_t *A, const i8_t *B_reo,
                       i32_t *C, int M, int K_r, int N_r,
                       i8_t *A_reo_pool, int num_threads) {
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

            i8_dispatch(my_A, B_reo, my_C, my_m, K_r, N_r,
                        my_A_reo, /*ldc_global=*/N_r);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// tile_N_i8 — N-tiling: split N columns across threads (for small M)
// ═══════════════════════════════════════════════════════════════════════
static void tile_N_i8(const i8_t *A, const i8_t *B_reo,
                       i32_t *C, int M, int K_r, int N_r,
                       int num_threads) {
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

            i8_t *my_A_reo = (i8_t *)aligned_alloc(64,
                (size_t)M * K_r * sizeof(i8_t));
            if (my_A_reo) {
                // B: each N-block = K_r * 8 int8 values (= K_r * 8 bytes)
                const i8_t *my_B = B_reo + (uint64_t)start_block * K_r * 8;
                // C: column offset; ldc_global = N_r (full row stride)
                i32_t *my_C = C + my_n_start;

                i8_dispatch(A, my_B, my_C, M, K_r, my_n,
                            my_A_reo, /*ldc_global=*/N_r);
                free(my_A_reo);
            }
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
    if (num_threads <= 0) num_threads = omp_get_max_threads();
    if (num_threads <= 0) num_threads = 1;

    int M_blocks = M / 8;

    if (M_blocks >= num_threads) {
        // M-tiling: one shared A_reorder pool, partitioned by M rows
        i8_t *A_reo_pool = (i8_t *)aligned_alloc(64,
            (size_t)M * K_r * sizeof(i8_t));
        if (!A_reo_pool) return;
        tile_M_i8(A, B_reo, C, M, K_r, N_r, A_reo_pool, num_threads);
        free(A_reo_pool);
    } else {
        // N-tiling: each thread allocates its own A_reorder
        tile_N_i8(A, B_reo, C, M, K_r, N_r, num_threads);
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
    i8_t *B_reo = (i8_t *)aligned_alloc(64,
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

    // Zero C padding columns
    if (N_r != N)
        for (int i = 0; i < M; i++)
            memset(C + i * N_r + N, 0, (N_r - N) * sizeof(i32_t));

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

    i8gemm_mt_dispatch(A_use, B_reo, C, M, K_r, N_r, num_threads);

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
                    f32_t *C, int m, int k, int n,
                    i8_t *A_reorder, int lda, int ldb, int ldc);
void i8gemm_k_ld1_f(const i8_t *A, const i8_t *B_reo,
                    f32_t *C, int m, int k, int n,
                    i8_t *A_reorder, int lda, int ldb, int ldc);
void i8gemm_k_ld2_f(const i8_t *A, const i8_t *B_reo,
                    f32_t *C, int m, int k, int n,
                    i8_t *A_reorder, int lda, int ldb, int ldc);
void i8gemm_k_ld4_f(const i8_t *A, const i8_t *B_reo,
                    f32_t *C, int m, int k, int n,
                    i8_t *A_reorder, int lda, int ldb, int ldc);

// ── i8_dispatch_f — Single-thread fp32 kernel dispatch ─────────────
static void i8_dispatch_f(const i8_t *A, const i8_t *B_reo,
                           f32_t *C, int m, int k, int n,
                           i8_t *A_reorder, int ldc_global) {
    int lda = k, ldb = k;
    int processed = 0;

    int m_full = (m / 8) * 8;
    if (m_full > 0) {
        i8gemm_k_ld_f(A, B_reo, C, m_full, k, n,
                      A_reorder, lda, ldb, ldc_global);
        processed = m_full;
    }

    int m_rem = m - processed;
    if (m_rem == 0) return;

    const i8_t *At = A + (uint64_t)processed * k;
    f32_t      *Ct = C + (uint64_t)processed * ldc_global;
    i8_t *A_reo_t  = A_reorder + (uint64_t)processed * k;

    if (m_rem >= 4) {
        i8gemm_k_ld4_f(At, B_reo, Ct, 4, k, n, A_reo_t, lda, ldb, ldc_global);
        processed += 4; m_rem -= 4;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * ldc_global;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 2) {
        i8gemm_k_ld2_f(At, B_reo, Ct, 2, k, n, A_reo_t, lda, ldb, ldc_global);
        processed += 2; m_rem -= 2;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * ldc_global;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 1) {
        i8gemm_k_ld1_f(At, B_reo, Ct, 1, k, n, A_reo_t, lda, ldb, ldc_global);
    }
}

// ── tile_M_i8_f — M-tiling for fp32 output ─────────────────────────
static void tile_M_i8_f(const i8_t *A, const i8_t *B_reo,
                         f32_t *C, int M, int K_r, int N_r,
                         i8_t *A_reo_pool, int num_threads) {
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

            i8_dispatch_f(my_A, B_reo, my_C, my_m, K_r, N_r,
                          my_A_reo, /*ldc_global=*/N_r);
        }
    }
}

// ── tile_N_i8_f — N-tiling for fp32 output ─────────────────────────
static void tile_N_i8_f(const i8_t *A, const i8_t *B_reo,
                         f32_t *C, int M, int K_r, int N_r,
                         int num_threads) {
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

            i8_t *my_A_reo = (i8_t *)aligned_alloc(64,
                (size_t)M * K_r * sizeof(i8_t));
            if (my_A_reo) {
                const i8_t *my_B = B_reo + (uint64_t)start_block * K_r * 8;
                f32_t *my_C = C + my_n_start;

                i8_dispatch_f(A, my_B, my_C, M, K_r, my_n,
                              my_A_reo, /*ldc_global=*/N_r);
                free(my_A_reo);
            }
        }
    }
}

// ── i8gemm_mt_dispatch_f — Core multi-threaded fp32 compute ────────
void i8gemm_mt_dispatch_f(const i8_t *A, const i8_t *B_reo,
                           f32_t *C, int M, int K_r, int N_r,
                           int num_threads) {
    if (num_threads <= 0) num_threads = omp_get_max_threads();
    if (num_threads <= 0) num_threads = 1;

    int M_blocks = M / 8;

    if (M_blocks >= num_threads) {
        i8_t *A_reo_pool = (i8_t *)aligned_alloc(64,
            (size_t)M * K_r * sizeof(i8_t));
        if (!A_reo_pool) return;
        tile_M_i8_f(A, B_reo, C, M, K_r, N_r, A_reo_pool, num_threads);
        free(A_reo_pool);
    } else {
        tile_N_i8_f(A, B_reo, C, M, K_r, N_r, num_threads);
    }
}

// ── i8gemm_mt_f — Convenience wrapper: pad, pack B, fp32 dispatch ──
void i8gemm_mt_f(const i8_t *A_orig, const i8_t *B_orig,
                  f32_t *C, int M, int K, int N,
                  int num_threads) {
    int K_r = ((K + 15) / 16) * 16;  if (K_r < 16) K_r = 16;
    int N_r = ((N + 7)  / 8)  * 8;   if (N_r < 8)   N_r = 8;

    // Pack B (same as i32 path)
    i8_t *B_reo = (i8_t *)aligned_alloc(64,
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

    // Zero C padding columns (f32)
    if (N_r != N)
        for (int i = 0; i < M; i++)
            memset(C + i * N_r + N, 0, (N_r - N) * sizeof(f32_t));

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

    i8gemm_mt_dispatch_f(A_use, B_reo, C, M, K_r, N_r, num_threads);

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
                          f32_t *C, int m, int k, int n,
                          i8_t *A_reorder, int lda, int ldb, int ldc,
                          const f32_t *bias);
void i8gemm_k_ld1_bias_f(const i8_t *A, const i8_t *B_reo,
                          f32_t *C, int m, int k, int n,
                          i8_t *A_reorder, int lda, int ldb, int ldc,
                          const f32_t *bias);
void i8gemm_k_ld2_bias_f(const i8_t *A, const i8_t *B_reo,
                          f32_t *C, int m, int k, int n,
                          i8_t *A_reorder, int lda, int ldb, int ldc,
                          const f32_t *bias);
void i8gemm_k_ld4_bias_f(const i8_t *A, const i8_t *B_reo,
                          f32_t *C, int m, int k, int n,
                          i8_t *A_reorder, int lda, int ldb, int ldc,
                          const f32_t *bias);

// ── i8_dispatch_bias_f — Single-thread bias kernel dispatch ───────────
static void i8_dispatch_bias_f(const i8_t *A, const i8_t *B_reo,
                                f32_t *C, int m, int k, int n,
                                i8_t *A_reorder, int ldc_global,
                                const f32_t *bias) {
    int lda = k, ldb = k;
    int processed = 0;

    int m_full = (m / 8) * 8;
    if (m_full > 0) {
        i8gemm_k_ld_bias_f(A, B_reo, C, m_full, k, n,
                           A_reorder, lda, ldb, ldc_global, bias);
        processed = m_full;
    }

    int m_rem = m - processed;
    if (m_rem == 0) return;

    const i8_t *At = A + (uint64_t)processed * k;
    f32_t      *Ct = C + (uint64_t)processed * ldc_global;
    i8_t *A_reo_t  = A_reorder + (uint64_t)processed * k;

    if (m_rem >= 4) {
        i8gemm_k_ld4_bias_f(At, B_reo, Ct, 4, k, n, A_reo_t,
                            lda, ldb, ldc_global, bias);
        processed += 4; m_rem -= 4;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * ldc_global;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 2) {
        i8gemm_k_ld2_bias_f(At, B_reo, Ct, 2, k, n, A_reo_t,
                            lda, ldb, ldc_global, bias);
        processed += 2; m_rem -= 2;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * ldc_global;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 1) {
        i8gemm_k_ld1_bias_f(At, B_reo, Ct, 1, k, n, A_reo_t,
                            lda, ldb, ldc_global, bias);
    }
}

// ── tile_M_i8_bias_f — M-tiling for bias fp32 output ──────────────────
static void tile_M_i8_bias_f(const i8_t *A, const i8_t *B_reo,
                              f32_t *C, int M, int K_r, int N_r,
                              i8_t *A_reo_pool, int num_threads,
                              const f32_t *bias) {
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

            i8_dispatch_bias_f(my_A, B_reo, my_C, my_m, K_r, N_r,
                               my_A_reo, /*ldc_global=*/N_r, bias);
        }
    }
}

// ── tile_N_i8_bias_f — N-tiling for bias fp32 output ──────────────────
static void tile_N_i8_bias_f(const i8_t *A, const i8_t *B_reo,
                              f32_t *C, int M, int K_r, int N_r,
                              int num_threads, const f32_t *bias) {
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

            i8_t *my_A_reo = (i8_t *)aligned_alloc(64,
                (size_t)M * K_r * sizeof(i8_t));
            if (my_A_reo) {
                const i8_t   *my_B   = B_reo + (uint64_t)start_block * K_r * 8;
                f32_t        *my_C   = C + my_n_start;
                const f32_t  *my_bias = bias + my_n_start;

                i8_dispatch_bias_f(A, my_B, my_C, M, K_r, my_n,
                                   my_A_reo, /*ldc_global=*/N_r, my_bias);
                free(my_A_reo);
            }
        }
    }
}

// ── i8gemm_mt_dispatch_bias_f — Core multi-threaded bias compute ──────
void i8gemm_mt_dispatch_bias_f(const i8_t *A, const i8_t *B_reo,
                                f32_t *C, int M, int K_r, int N_r,
                                int num_threads, const f32_t *bias) {
    if (num_threads <= 0) num_threads = omp_get_max_threads();
    if (num_threads <= 0) num_threads = 1;

    int M_blocks = M / 8;

    if (M_blocks >= num_threads) {
        i8_t *A_reo_pool = (i8_t *)aligned_alloc(64,
            (size_t)M * K_r * sizeof(i8_t));
        if (!A_reo_pool) return;
        tile_M_i8_bias_f(A, B_reo, C, M, K_r, N_r, A_reo_pool,
                          num_threads, bias);
        free(A_reo_pool);
    } else {
        tile_N_i8_bias_f(A, B_reo, C, M, K_r, N_r, num_threads, bias);
    }
}

// ── i8gemm_mt_bias_f — Convenience wrapper: pad, pack B, bias dispatch ─
void i8gemm_mt_bias_f(const i8_t *A_orig, const i8_t *B_orig,
                       f32_t *C, int M, int K, int N,
                       int num_threads, const f32_t *bias) {
    int K_r = ((K + 15) / 16) * 16;  if (K_r < 16) K_r = 16;
    int N_r = ((N + 7)  / 8)  * 8;   if (N_r < 8)   N_r = 8;

    // Pack B (same as other i8 variants)
    i8_t *B_reo = (i8_t *)aligned_alloc(64,
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

    i8gemm_mt_dispatch_bias_f(A_use, B_reo, C, M, K_r, N_r,
                               num_threads, bias_use);

    free(A_pad);
    free(bias_pad);
    free(B_reo);
}
