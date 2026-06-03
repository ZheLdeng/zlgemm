// bf16gemm_mt.c — OpenMP Multi-threaded BF16 GEMM (library)
//
// Auto-selects M-tiling or N-tiling based on shape.
// B must be pre-packed by caller (bf16_pack_B) outside the timed region.
//
// Exported API:
//   void bf16_pack_B(const bf16_t *B, bf16_t *B_reo, int K, int N);
//   void bf16gemm_mt_dispatch(const bf16_t *A, const bf16_t *B_reo,
//                              f32_t *C, int M, int K_r, int N_r, int nthreads);
//   void bf16gemm_mt(const bf16_t *A, const bf16_t *B,
//                     f32_t *C, int M, int K, int N, int nthreads);
//
// Build:
//   cc -c bf16gemm_mt.c -march=armv8.6-a+bf16 -O2 -Wall -fopenmp
//   (link with bf16gemm_k.S and -fopenmp -lm)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>

// ═══════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════
typedef uint16_t bf16_t;
typedef float    f32_t;

// ═══════════════════════════════════════════════════════════════════════
// Kernel declarations (from bf16gemm_k.S)
// ═══════════════════════════════════════════════════════════════════════
void bf16gemm_k_ld (const bf16_t *A, const bf16_t *B_reo,
                    f32_t *C, int m, int k, int n,
                    bf16_t *A_reorder, int lda, int ldb, int ldc);
void bf16gemm_k_ld1(const bf16_t *A, const bf16_t *B_reo,
                    f32_t *C, int m, int k, int n,
                    bf16_t *A_reorder, int lda, int ldb, int ldc);
void bf16gemm_k_ld2(const bf16_t *A, const bf16_t *B_reo,
                    f32_t *C, int m, int k, int n,
                    bf16_t *A_reorder, int lda, int ldb, int ldc);
void bf16gemm_k_ld4(const bf16_t *A, const bf16_t *B_reo,
                    f32_t *C, int m, int k, int n,
                    bf16_t *A_reorder, int lda, int ldb, int ldc);

// ═══════════════════════════════════════════════════════════════════════
// bf16_pack_B — Pack B matrix for bfmmla kernel (caller allocates B_reo)
//
// B_reo size: K * N bf16 elements.
// Layout (outer→inner):
//   for each N-block (8 cols):
//     for each K-block (4 rows):
//       for each of 4 column-pairs: 4 K-rows of c0, then 4 K-rows of c1
//
// One N-block = K * 8 bf16 values = K * 16 bytes.
// ═══════════════════════════════════════════════════════════════════════
void bf16_pack_B(const bf16_t *B, bf16_t *B_reo, int K, int N) {
    int idx = 0;
    for (int cb = 0; cb < N / 8; ++cb) {
        for (int rb = 0; rb < K / 4; ++rb) {
            int row_base = rb * 4, col_base = cb * 8;
            for (int cp = 0; cp < 4; ++cp) {
                int c0 = col_base + cp * 2, c1 = c0 + 1;
                for (int i = 0; i < 4; ++i)
                    B_reo[idx++] = B[(row_base + i) * N + c0];
                for (int i = 0; i < 4; ++i)
                    B_reo[idx++] = B[(row_base + i) * N + c1];
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// bf16_dispatch — Single-thread kernel dispatch (internal)
//
// Handles arbitrary M by decomposing into full-8 + tail-4/2/1 calls.
// ldc_global: C row stride in f32 elements (may differ from n for N-tiling).
// ═══════════════════════════════════════════════════════════════════════
static void bf16_dispatch(const bf16_t *A, const bf16_t *B_reo,
                           f32_t *C, int m, int k, int n,
                           bf16_t *A_reorder, int ldc_global) {
    int lda = k, ldb = k;
    int processed = 0;

    int m_full = (m / 8) * 8;
    if (m_full > 0) {
        bf16gemm_k_ld(A, B_reo, C, m_full, k, n,
                      A_reorder, lda, ldb, ldc_global);
        processed = m_full;
    }

    int m_rem = m - processed;
    if (m_rem == 0) return;

    const bf16_t *At = A + (uint64_t)processed * k;
    f32_t        *Ct = C + (uint64_t)processed * ldc_global;
    bf16_t *A_reo_t  = A_reorder + (uint64_t)processed * k;

    if (m_rem >= 4) {
        bf16gemm_k_ld4(At, B_reo, Ct, 4, k, n, A_reo_t, lda, ldb, ldc_global);
        processed += 4; m_rem -= 4;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * ldc_global;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 2) {
        bf16gemm_k_ld2(At, B_reo, Ct, 2, k, n, A_reo_t, lda, ldb, ldc_global);
        processed += 2; m_rem -= 2;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * ldc_global;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 1) {
        bf16gemm_k_ld1(At, B_reo, Ct, 1, k, n, A_reo_t, lda, ldb, ldc_global);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// tile_M — M-tiling: split M rows across threads
// ═══════════════════════════════════════════════════════════════════════
static void tile_M(const bf16_t *A, const bf16_t *B_reo,
                    f32_t *C, int M, int K_r, int N_r,
                    bf16_t *A_reo_pool, int num_threads) {
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
            const bf16_t *my_A = A + (uint64_t)my_m_start * K_r;
            f32_t        *my_C = C + (uint64_t)my_m_start * N_r;
            bf16_t *my_A_reo  = A_reo_pool + (uint64_t)my_m_start * K_r;

            bf16_dispatch(my_A, B_reo, my_C, my_m, K_r, N_r,
                          my_A_reo, /*ldc_global=*/N_r);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// tile_N — N-tiling: split N columns across threads (for small M)
//
// Each thread allocates its own A_reorder because each thread's "first N"
// iteration for each M-block writes A_reorder independently.
// B is shared read-only; each thread offsets into its N-block range.
// ═══════════════════════════════════════════════════════════════════════
static void tile_N(const bf16_t *A, const bf16_t *B_reo,
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

            bf16_t *my_A_reo = (bf16_t *)aligned_alloc(64,
                (size_t)M * K_r * sizeof(bf16_t));
            if (my_A_reo) {
                // B: each N-block = K_r * 8 bf16 values
                const bf16_t *my_B = B_reo + (uint64_t)start_block * K_r * 8;
                // C: column offset; ldc_global = N_r (full row stride)
                f32_t *my_C = C + my_n_start;

                bf16_dispatch(A, my_B, my_C, M, K_r, my_n,
                              my_A_reo, /*ldc_global=*/N_r);
                free(my_A_reo);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// bf16gemm_mt_dispatch — Core multi-threaded compute (no allocation, no pack)
//
// A:   M × K_r  row-major bf16 (padded to K_r multiple of 8)
// B_reo: pre-packed by bf16_pack_B, K_r × N_r bf16
// C:   M × N_r  row-major f32 (output, zero-init before call)
// K_r, N_r: must be multiples of 8
//
// This is the function to put inside the timed benchmark region.
// ═══════════════════════════════════════════════════════════════════════
void bf16gemm_mt_dispatch(const bf16_t *A, const bf16_t *B_reo,
                           f32_t *C, int M, int K_r, int N_r,
                           int num_threads) {
    if (num_threads <= 0) num_threads = omp_get_max_threads();
    if (num_threads <= 0) num_threads = 1;

    int M_blocks = M / 8;

    if (M_blocks >= num_threads) {
        // M-tiling: one shared A_reorder pool, partitioned by M rows
        bf16_t *A_reo_pool = (bf16_t *)aligned_alloc(64,
            (size_t)M * K_r * sizeof(bf16_t));
        if (!A_reo_pool) return;
        tile_M(A, B_reo, C, M, K_r, N_r, A_reo_pool, num_threads);
        free(A_reo_pool);
    } else {
        // N-tiling: each thread allocates its own A_reorder
        tile_N(A, B_reo, C, M, K_r, N_r, num_threads);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// bf16gemm_mt — Convenience wrapper: pad, pack B, then dispatch
//
// A_orig: M×K row-major bf16 (K need not be multiple of 8)
// B_orig: K×N row-major bf16 (K,N need not be multiples of 8)
// C:      M×N row-major f32 output (zero-init before call)
//
// Packs B internally — for repeated calls with the same B, prefer
// bf16_pack_B + bf16gemm_mt_dispatch to avoid re-packing.
// ═══════════════════════════════════════════════════════════════════════
void bf16gemm_mt(const bf16_t *A_orig, const bf16_t *B_orig,
                  f32_t *C, int M, int K, int N,
                  int num_threads) {
    int K_r = ((K + 7) / 8) * 8;   if (K_r < 8)  K_r = 8;
    int N_r = ((N + 7) / 8) * 8;   if (N_r < 8)  N_r = 8;

    // Pack B
    bf16_t *B_reo = (bf16_t *)aligned_alloc(64,
        (size_t)K_r * N_r * sizeof(bf16_t));
    if (!B_reo) return;

    bf16_t *B_pad = NULL;
    const bf16_t *B_use;
    if (K_r == K && N_r == N) {
        B_use = B_orig;
    } else {
        B_pad = (bf16_t *)calloc((size_t)K_r * N_r, sizeof(bf16_t));
        if (!B_pad) { free(B_reo); return; }
        for (int i = 0; i < K; i++)
            memcpy(B_pad + i * N_r, B_orig + i * N, N * sizeof(bf16_t));
        B_use = B_pad;
    }
    bf16_pack_B(B_use, B_reo, K_r, N_r);
    free(B_pad);

    // Zero C padding columns
    if (N_r != N)
        for (int i = 0; i < M; i++)
            memset(C + i * N_r + N, 0, (N_r - N) * sizeof(f32_t));

    // Pad A if needed
    bf16_t *A_pad = NULL;
    const bf16_t *A_use;
    if (K_r == K) {
        A_use = A_orig;
    } else {
        A_pad = (bf16_t *)calloc((size_t)M * K_r, sizeof(bf16_t));
        if (!A_pad) { free(B_reo); return; }
        for (int i = 0; i < M; i++)
            memcpy(A_pad + i * K_r, A_orig + i * K, K * sizeof(bf16_t));
        A_use = A_pad;
    }

    bf16gemm_mt_dispatch(A_use, B_reo, C, M, K_r, N_r, num_threads);

    free(A_pad);
    free(B_reo);
}
