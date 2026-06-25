// i8gemm_msplit.c -- M-split-focused multithreaded I8 GEMM dispatch.
//
// See i8gemm_msplit.h for the contract. Two regimes:
//   * M-split    (k_slices == 1): each thread owns a contiguous 8-row-aligned
//                M-band and computes the full N. No N subdivision, no reduction,
//                best inner locality. Used whenever M-banding alone can fill the
//                requested threads (mblk = ceil(M/8) >= P).
//   * split-K    (k_slices  > 1): when M is too small to fill P threads by
//                M-banding, the K reduction is split into k_slices ranges so the
//                worker grid is (m_groups x k_slices) = P busy threads. Each
//                thread writes a private partial; a parallel reduction sums them.
//                The 8-row tile stays full-width (the 1/2/4-row kernels waste
//                compute -- they run the full 8-row smmla machinery -- so finer
//                M granularity loses; split-K preserves 8-row efficiency).
//
// The microkernel i8gemm_msk is i8gemm_k_hybrid with the B N-panel stride taken
// from params.ldb, so a K-slice (params.k) can sweep the full N in one call while
// B_reo keeps its full-K layout.

#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "gemm_params.h"
#include "i8gemm_msplit.h"

// From i8gemm_sve.c (same B layout). Declared here to avoid a header dependency.
void i8_pack_B(const i8_t *B, i8_t *B_reo, int K, int N);
// The tuned single-thread dispatch (auto-selects packed/m12/hybrid microkernel).
// Reused per M-band so the M-split path matches the main lib's per-core kernel.
void i8gemm_mt_dispatch(const i8_t *A, const i8_t *B_reo, i32_t *C,
                        int M, int K_r, int N_r, int nthreads);

// Split-K microkernel (i8gemm_msplit_k.S).
void i8gemm_msk(const i8_t *A, const i8_t *B_reo, i32_t *C,
                i8_t *unused, const gemm_params_t *p);

static size_t msp_round_up_64(size_t s) { return (s + 63u) & ~(size_t)63u; }

static int msp_n_tile(void) {
    // VL bytes / 2  (== i8_pack_B's segs*8). 16 for VL=256.
    static int cached = 0;
    if (!cached) {
        long vlb = 0;
        __asm__("cntb %0" : "=r"(vlb));
        cached = (int)(vlb / 2);
    }
    return cached;
}

// Thread-local cached scratch for the split-K partials (owned by the master
// thread; workers only write disjoint regions of a captured pointer).
static __thread i32_t *g_part = NULL;
static __thread size_t g_part_cap = 0;

static i32_t *msp_partials(size_t n_i32) {
    size_t bytes = n_i32 * sizeof(i32_t);
    if (bytes > g_part_cap) {
        free(g_part);
        g_part = (i32_t *)aligned_alloc(64, msp_round_up_64(bytes));
        g_part_cap = g_part ? msp_round_up_64(bytes) : 0;
    }
    return g_part;
}

// Decide the worker grid (m_groups x k_slices). Honours num_threads, never
// subdivides N. Returns the two factors; product <= effective threads.
static void msp_pick_grid(int M, int K_r, int N_r, int P,
                          int *out_mg, int *out_ks) {
    if (P <= 0) P = omp_get_max_threads();
    if (P <= 0) P = 1;

    const int n_tile = msp_n_tile();
    const int n_tiles = N_r / n_tile;
    const int mblk = (M + 7) / 8;          // 8-row blocks
    const int kblk = K_r / 8;              // 8-deep K blocks (split granularity)

    // Light fork-amortisation: keep each worker above a small mac floor so we
    // don't spawn threads for negligible work. Tuned low (256K macs) -- the
    // split-K path has a single cheap fork, unlike the packed A-pack region.
    long total_macs = (long)M * (long)K_r * (long)N_r;
    int mac_cap = (int)(total_macs / (256L << 10));
    if (mac_cap < 1) mac_cap = 1;
    if (P > mac_cap) P = mac_cap;

    int mg = mblk < P ? mblk : P;
    if (mg < 1) mg = 1;
    int ks = (P + mg - 1) / mg;            // ceil: fill the rest with K-slices
    if (ks > kblk) ks = kblk;
    if (ks < 1) ks = 1;
    // A K-slice should be worth a fork; require >= 2 K-blocks (128 deep) each,
    // else prefer fewer, fatter slices.
    while (ks > 1 && kblk / ks < 2) ks--;
    // Debug/experiment knobs: I8MS_NOSPLITK=1 forces pure M-split (ks=1, the
    // "naive M-split-only" baseline); I8MS_KS=<n> pins the K-slice count.
    const char *nosk = getenv("I8MS_NOSPLITK");
    if (nosk && atoi(nosk)) ks = 1;
    const char *ksenv = getenv("I8MS_KS");
    if (ksenv) { int v = atoi(ksenv); if (v >= 1 && v <= kblk) ks = v; }
    *out_mg = mg;
    *out_ks = ks;
    (void)n_tiles;
}

void i8gemm_msplit_dispatch(const i8_t *A, const i8_t *B_reo,
                            i32_t *C, int M, int K_r, int N_r,
                            int num_threads) {
    if (M <= 0 || K_r <= 0 || N_r <= 0) return;
    const int n_tile = msp_n_tile();
    const size_t ldb = (size_t)K_r * (size_t)n_tile;   // bytes between N-panels
    const int mblk = (M + 7) / 8;
    const int kblk = K_r / 8;

    int mg, ks;
    msp_pick_grid(M, K_r, N_r, num_threads, &mg, &ks);

    if (ks <= 1) {
        // Pure M-split: contiguous 8-row-aligned bands, one per group. Each band
        // is computed by the tuned single-thread dispatch so we inherit the best
        // per-core microkernel (packed / m12 / hybrid) for the band's shape.
        #pragma omp parallel num_threads(mg)
        {
            int tid = omp_get_thread_num();
            int bq = mblk / mg, br = mblk % mg;
            int b0 = tid * bq + (tid < br ? tid : br);
            int nb = bq + (tid < br ? 1 : 0);
            int m0 = b0 * 8;
            int m1 = (b0 + nb) * 8;
            if (m1 > M) m1 = M;
            if (m0 < m1)
                i8gemm_mt_dispatch(A + (size_t)m0 * K_r, B_reo,
                                   C + (size_t)m0 * N_r, m1 - m0, K_r, N_r, 1);
        }
        (void)ldb;
        return;
    }

    // split-K: (mg x ks) worker grid, private partials, then reduce -- all in a
    // SINGLE parallel region (one fork per call). Each thread computes its
    // (M-band x K-slice) partial; after a barrier, the ki==0 thread of each
    // M-band sums that band's ks partials into C.
    i32_t *part = msp_partials((size_t)ks * (size_t)M * (size_t)N_r);
    if (!part) {
        gemm_params_t p = {M, K_r, N_r, K_r, (int)ldb, N_r};
        i8gemm_msk(A, B_reo, C, NULL, &p);
        return;
    }
    const size_t slice_elems = (size_t)M * (size_t)N_r;

    #pragma omp parallel num_threads(mg * ks)
    {
        int tid = omp_get_thread_num();
        int mi = tid % mg;     // M-band index
        int ki = tid / mg;     // K-slice index

        int bq = mblk / mg, br = mblk % mg;
        int b0 = mi * bq + (mi < br ? mi : br);
        int nb = bq + (mi < br ? 1 : 0);
        int m0 = b0 * 8;
        int m1 = (b0 + nb) * 8;
        if (m1 > M) m1 = M;

        int kq = kblk / ks, kr = kblk % ks;
        int kb0 = ki * kq + (ki < kr ? ki : kr);
        int kbn = kq + (ki < kr ? 1 : 0);
        int k0 = kb0 * 8;
        int klen = kbn * 8;

        if (m0 < m1 && klen > 0) {
            i32_t *Cp = part + (size_t)ki * slice_elems;
            gemm_params_t p = {m1 - m0, klen, N_r, K_r, (int)ldb, N_r};
            i8gemm_msk(A + (size_t)m0 * K_r + (size_t)k0,
                       B_reo + (size_t)k0 * n_tile,
                       Cp + (size_t)m0 * N_r, NULL, &p);
        }

        #pragma omp barrier

        // Reduce all ks partials into C, work split across the whole team
        // (partition the flat M*N_r output). Critical for small M where only a
        // few M-bands exist but many K-slices contribute.
        if (ks > 1) {
            const long total = (long)M * (long)N_r;
            const int P = mg * ks;
            long chunk = (total + P - 1) / P;
            long o0 = (long)tid * chunk;
            long o1 = o0 + chunk;
            if (o1 > total) o1 = total;
            for (long o = o0; o < o1; o++) {
                i32_t acc = part[o];
                for (int s = 1; s < ks; s++)
                    acc += part[(size_t)s * slice_elems + o];
                C[o] = acc;
            }
        } else if (m0 < m1) {
            for (long o = (long)m0 * N_r; o < (long)m1 * N_r; o++)
                C[o] = part[o];
        }
    }
    (void)kblk;
}

// ---- convenience wrapper (pad + pack) ----

static int msp_round_up_int(int x, int q) { return ((x + q - 1) / q) * q; }

void i8gemm_msplit(const i8_t *A_orig, const i8_t *B_orig,
                   i32_t *C, int M, int K, int N, int num_threads) {
    const int n_tile = msp_n_tile();
    int K_r = msp_round_up_int(K < 16 ? 16 : K, 16);
    int N_r = msp_round_up_int(N < n_tile ? n_tile : N, n_tile);

    i8_t *B_pad = (i8_t *)calloc((size_t)K_r * N_r, 1);
    i8_t *B_reo = (i8_t *)aligned_alloc(64, msp_round_up_64((size_t)K_r * N_r));
    if (!B_pad || !B_reo) { free(B_pad); free(B_reo); return; }
    for (int i = 0; i < K; i++)
        memcpy(B_pad + (size_t)i * N_r, B_orig + (size_t)i * N, (size_t)N);
    i8_pack_B(B_pad, B_reo, K_r, N_r);
    free(B_pad);

    i8_t *A_use = (i8_t *)A_orig;
    if (K_r != K) {
        A_use = (i8_t *)calloc((size_t)M * K_r, 1);
        if (!A_use) { free(B_reo); return; }
        for (int i = 0; i < M; i++)
            memcpy(A_use + (size_t)i * K_r, A_orig + (size_t)i * K, (size_t)K);
    }

    i32_t *C_pad = (i32_t *)calloc((size_t)M * N_r, sizeof(i32_t));
    if (C_pad) {
        i8gemm_msplit_dispatch(A_use, B_reo, C_pad, M, K_r, N_r, num_threads);
        for (int i = 0; i < M; i++)
            memcpy(C + (size_t)i * N, C_pad + (size_t)i * N_r,
                   (size_t)N * sizeof(i32_t));
        free(C_pad);
    }
    if (A_use != A_orig) free(A_use);
    free(B_reo);
}
