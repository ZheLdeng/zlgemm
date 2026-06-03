// calculate_bf16mm_tail.c
// Unified GEMM: dispatches M to bf16gemm_k_ld (full 8-row blocks) +
// bf16gemm_k_ld1/ld2/ld4 (tail 1/2/4 rows).
// m,n,k == lda,ldb,ldc (contiguous matrices).
//
// Build:
//   cc -o bf16_tail_test calculate_bf16mm_tail.c bf16gemm_k_ld.S bf16gemm_k_tail.S \
//      -march=armv8.2-a+bf16 -O2 -Wall -lm
//
// Usage:
//   ./bf16_tail_test M K N         # single size, correctness + perf
//   ./bf16_tail_test sweep         # correctness sweep over many sizes
//   ./bf16_tail_test csv shape.csv # benchmark all shapes from CSV
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>

typedef float    AccumType;
typedef uint16_t BF16Type;

#define MATRIX_EPSILON 0.5f   // tolerance for bf16 float accumulation errors

// ── bf16 conversion helpers ──────────────────────────────────────
static inline BF16Type float_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    uint32_t rounding = ((u >> 16) & 1) + 0x7FFF;
    u += rounding;
    return (BF16Type)(u >> 16);
}

static inline float bf16_to_float(BF16Type b) {
    uint32_t u = ((uint32_t)b) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

// ── Timing ───────────────────────────────────────────────────────
static double get_time(struct timespec *start, struct timespec *end) {
    return end->tv_sec - start->tv_sec +
        (end->tv_nsec - start->tv_nsec) * 1e-9;
}

// ── Kernel declarations ──────────────────────────────────────────
void bf16gemm_k_ld (const BF16Type *A, const BF16Type *B_reordered,
                    AccumType *C, int m, int k, int n,
                    BF16Type *A_reorder, int lda, int ldb, int ldc);
void bf16gemm_k_ld1(const BF16Type *A, const BF16Type *B_reordered,
                    AccumType *C, int m, int k, int n,
                    BF16Type *A_reorder, int lda, int ldb, int ldc);
void bf16gemm_k_ld2(const BF16Type *A, const BF16Type *B_reordered,
                    AccumType *C, int m, int k, int n,
                    BF16Type *A_reorder, int lda, int ldb, int ldc);
void bf16gemm_k_ld4(const BF16Type *A, const BF16Type *B_reordered,
                    AccumType *C, int m, int k, int n,
                    BF16Type *A_reorder, int lda, int ldb, int ldc);

// ── B reorder for bf16: N-major, K/4 blocks, column-major 4×2 ──
static void reorder_B_bf16(const BF16Type* B, BF16Type* B_reo,
                            int K, int N) {
    assert(K % 4 == 0);
    assert(N % 8 == 0);
    int idx = 0;
    for (int cb = 0; cb < N/8; ++cb) {
        for (int rb = 0; rb < K/4; ++rb) {
            int row_base = rb * 4;
            int col_base = cb * 8;
            for (int cp = 0; cp < 4; ++cp) {
                int c0 = col_base + cp * 2;
                int c1 = c0 + 1;
                for (int i = 0; i < 4; ++i)
                    B_reo[idx++] = B[(row_base + i) * N + c0];
                for (int i = 0; i < 4; ++i)
                    B_reo[idx++] = B[(row_base + i) * N + c1];
            }
        }
    }
}

// ── Matrix generation ────────────────────────────────────────────
static BF16Type* bf16_generate(int rows, int cols) {
    BF16Type* m = (BF16Type*)malloc(rows * cols * sizeof(BF16Type));
    assert(m);
    for (int i = 0; i < rows * cols; i++) {
        float val = ((float)rand() / (float)RAND_MAX) * 4.0f - 2.0f;
        m[i] = float_to_bf16(val);
    }
    return m;
}

static AccumType* float32_zero(int rows, int cols) {
    return (AccumType*)calloc(rows * cols, sizeof(AccumType));
}

// ── Reference GEMM (bf16→float accumulation) ─────────────────────
static void bf16_ref_gemm(const BF16Type* A, const BF16Type* B,
                          AccumType* C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += bf16_to_float(A[i * k + l]) *
                       bf16_to_float(B[l * n + j]);
            }
            C[i * n + j] += sum;
        }
    }
}

// ── Verification ─────────────────────────────────────────────────
static int verify(const BF16Type* A, const BF16Type* B_orig,
                  const AccumType* C_result, int m, int k, int n) {
    AccumType* ref = (AccumType*)calloc(m * n, sizeof(AccumType));
    bf16_ref_gemm(A, B_orig, ref, m, k, n);

    int ok = 1;
    float max_err = 0.0f;
    int first_i = -1, first_j = -1;
    float first_got = 0, first_exp = 0;

    for (int i = 0; i < m && ok; i++) {
        for (int j = 0; j < n && ok; j++) {
            float err = fabsf(C_result[i * n + j] - ref[i * n + j]);
            if (err > max_err) max_err = err;
            if (err > MATRIX_EPSILON && ok) {
                first_i = i; first_j = j;
                first_got = C_result[i * n + j];
                first_exp = ref[i * n + j];
                ok = 0;
            }
        }
    }

    if (ok) {
        printf("  Max error: %e\n", (double)max_err);
    } else {
        printf("  FIRST MISMATCH at (%d,%d): got=%f expect=%f err=%e\n",
               first_i, first_j, (double)first_got, (double)first_exp,
               (double)fabsf(first_got - first_exp));
        // Show first few mismatches for debugging
        int shown = 1;
        for (int i = 0; i < m && shown < 5; i++)
            for (int j = 0; j < n && shown < 5; j++) {
                float err = fabsf(C_result[i * n + j] - ref[i * n + j]);
                if (err > MATRIX_EPSILON) {
                    printf("  MISMATCH at (%d,%d): got=%f expect=%f err=%e\n",
                           i, j, (double)C_result[i * n + j],
                           (double)ref[i * n + j], (double)err);
                    shown++;
                }
            }
    }

    free(ref);
    return ok;
}

// ═══════════════════════════════════════════════════════════════════
// Unified GEMM dispatch: M → 8-row blocks (main) + 4/2/1-row tail
// ═══════════════════════════════════════════════════════════════════
static void bf16gemm_k_dispatch(const BF16Type *A,
                                 const BF16Type *B_reordered,
                                 AccumType *C, int m, int k, int n,
                                 BF16Type *A_reorder) {
    int lda = k, ldb = k, ldc = n;
    int processed = 0;

    // ── Full 8-row blocks ──
    int m_full = (m / 8) * 8;
    if (m_full > 0) {
        bf16gemm_k_ld(A, B_reordered, C, m_full, k, n,
                      A_reorder, lda, ldb, ldc);
        processed = m_full;
    }

    int m_rem = m - processed;
    if (m_rem == 0) return;

    // Advance pointers to the tail section
    const BF16Type *At = A      + (uint64_t)processed * k;
    AccumType       *Ct = C     + (uint64_t)processed * n;
    BF16Type *A_reo_t  = A_reorder + (uint64_t)processed * k;

    // ── Tail: 4-row block ──
    if (m_rem >= 4) {
        bf16gemm_k_ld4(At, B_reordered, Ct, 4, k, n,
                       A_reo_t, lda, ldb, ldc);
        processed += 4;
        m_rem    -= 4;
        At        = A      + (uint64_t)processed * k;
        Ct        = C     + (uint64_t)processed * n;
        A_reo_t   = A_reorder + (uint64_t)processed * k;
    }

    // ── Tail: 2-row block ──
    if (m_rem >= 2) {
        bf16gemm_k_ld2(At, B_reordered, Ct, 2, k, n,
                       A_reo_t, lda, ldb, ldc);
        processed += 2;
        m_rem    -= 2;
        At        = A      + (uint64_t)processed * k;
        Ct        = C     + (uint64_t)processed * n;
        A_reo_t   = A_reorder + (uint64_t)processed * k;
    }

    // ── Tail: 1-row block ──
    if (m_rem >= 1) {
        bf16gemm_k_ld1(At, B_reordered, Ct, 1, k, n,
                       A_reo_t, lda, ldb, ldc);
        processed += 1;
    }
}

// ── Single-size test ─────────────────────────────────────────────
static int test_single(int m, int k, int n) {
    // Round K,N up for kernel compatibility (K multiple of 8, N multiple of 8)
    int k_r = ((k + 7) / 8) * 8;
    if (k_r < 8) k_r = 8;
    int n_r = ((n + 7) / 8) * 8;
    if (n_r < 8) n_r = 8;

    srand(42);
    BF16Type *A      = bf16_generate(m, k_r);
    BF16Type *B_orig = bf16_generate(k_r, n_r);
    BF16Type *B_reo  = (BF16Type*)malloc((size_t)k_r * n_r * sizeof(BF16Type));
    reorder_B_bf16(B_orig, B_reo, k_r, n_r);

    AccumType *C = float32_zero(m, n_r);
    // A_reorder: allocate (m + 8) * k_r bf16 for margin
    BF16Type *A_reo = (BF16Type*)malloc((size_t)(m + 8) * k_r * sizeof(BF16Type));

    bf16gemm_k_dispatch(A, B_reo, C, m, k_r, n_r, A_reo);

    int ok = verify(A, B_orig, C, m, k_r, n_r);

    free(A); free(B_orig); free(B_reo); free(C); free(A_reo);
    return ok;
}

// ── Benchmark with timing ────────────────────────────────────────
static void bench_single(int m, int k, int n, int loops) {
    int k_r = ((k + 7) / 8) * 8;
    if (k_r < 8) k_r = 8;
    int n_r = ((n + 7) / 8) * 8;
    if (n_r < 8) n_r = 8;

    srand(42);
    BF16Type *A      = bf16_generate(m, k_r);
    BF16Type *B_orig = bf16_generate(k_r, n_r);
    BF16Type *B_reo  = (BF16Type*)malloc((size_t)k_r * n_r * sizeof(BF16Type));
    reorder_B_bf16(B_orig, B_reo, k_r, n_r);

    AccumType *C = float32_zero(m, n_r);
    BF16Type *A_reo = (BF16Type*)malloc((size_t)(m + 8) * k_r * sizeof(BF16Type));

    // Warmup
    for (int w = 0; w < 3; w++)
        bf16gemm_k_dispatch(A, B_reo, C, m, k_r, n_r, A_reo);

    struct timespec s, e;
    clock_gettime(CLOCK_MONOTONIC_RAW, &s);
    for (int i = 0; i < loops; i++)
        bf16gemm_k_dispatch(A, B_reo, C, m, k_r, n_r, A_reo);
    clock_gettime(CLOCK_MONOTONIC_RAW, &e);

    double t = get_time(&s, &e);
    double t_us = t * 1e6 / loops;
    double gflops = (2.0 * m * k_r * n_r) / (t / loops) * 1e-9;

    printf("%5d %5d %5d  %5d %5d %5d %5d  %8.1f  %9.1f\n",
           m, k, n, m, k_r, n_r, loops, t_us, gflops);

    // Verify correctness
    int ok = verify(A, B_orig, C, m, k_r, n_r);
    if (!ok) printf("  *** CORRECTNESS FAILED ***\n");

    free(A); free(B_orig); free(B_reo); free(C); free(A_reo);
}

// ── Auto-tune loops ──────────────────────────────────────────────
static int auto_loops(int m, int k, int n) {
    double flops = 2.0 * m * k * n;
    int loops = (int)(50e-3 * 300e9 / flops);  // ~50ms target at ~300 GFLOPS
    if (loops < 1)  loops = 1;
    if (loops > 500) loops = 500;
    return loops;
}

// ═══════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════
int main(int argc, char *argv[]) {
    if (argc == 2 && strcmp(argv[1], "sweep") == 0) {
        // ── Correctness sweep ──
        printf("=== BF16 Tail Correctness Sweep ===\n");
        int M_sizes[] = {1,2,3,4,5,6,7, 8,9,10,11,12,13,14,15,
                         16,17,18,19,20,21,22,23,
                         24,31,32,33,39,40,41,47,48,49,55,56,57,63,64,65};
        int K_sizes[] = {8, 16, 24, 32, 48, 64, 80, 128};
        int N_sizes[] = {8, 16, 24, 32, 64};
        int failed = 0, total = 0;
        for (int mi = 0; mi < sizeof(M_sizes)/sizeof(M_sizes[0]); mi++) {
            for (int ki = 0; ki < sizeof(K_sizes)/sizeof(K_sizes[0]); ki++) {
                for (int ni = 0; ni < sizeof(N_sizes)/sizeof(N_sizes[0]); ni++) {
                    total++;
                    int _m = M_sizes[mi], _k = K_sizes[ki], _n = N_sizes[ni];
                    if (!test_single(_m, _k, _n)) {
                        printf("FAIL: M=%d K=%d N=%d\n", _m, _k, _n);
                        failed++;
                    }
                }
            }
        }
        printf("Results: %d/%d passed\n", total - failed, total);
        return failed > 0 ? 1 : 0;

    } else if (argc == 3 && strcmp(argv[1], "csv") == 0) {
        // ── Benchmark from shape.csv ──
        FILE *fp = fopen(argv[2], "r");
        if (!fp) { fprintf(stderr, "Cannot open %s\n", argv[2]); return 1; }

        printf("# M_orig K_orig N_orig  M_r  K_r  N_r loops  t_us  GFLOPS\n");
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            if (line[0] < '0' || line[0] > '9') continue;
            int m, k, n;
            if (sscanf(line, "%d,%d,%d", &m, &k, &n) != 3) continue;
            if (m <= 0 || k <= 0 || n <= 0) continue;

            int k_r = ((k + 7) / 8) * 8;
            if (k_r < 8) k_r = 8;
            int n_r = ((n + 7) / 8) * 8;
            if (n_r < 8) n_r = 8;
            int loops = auto_loops(m, k_r, n_r);
            bench_single(m, k, n, loops);
        }
        fclose(fp);
        return 0;
    }

    // ── Single-size test ──
    int m = (argc >= 4) ? atoi(argv[1]) : 16;
    int k = (argc >= 4) ? atoi(argv[2]) : 16;
    int n = (argc >= 4) ? atoi(argv[3]) : 8;

    // Round up for kernel
    int k_r = ((k + 7) / 8) * 8;
    if (k_r < 8) k_r = 8;
    int n_r = ((n + 7) / 8) * 8;
    if (n_r < 8) n_r = 8;

    printf("BF16 Tail Test: m=%d, k=%d -> k_r=%d, n=%d -> n_r=%d\n",
           m, k, k_r, n, n_r);

    srand(time(NULL));
    BF16Type *A      = bf16_generate(m, k_r);
    BF16Type *B_orig = bf16_generate(k_r, n_r);
    BF16Type *B_reo  = (BF16Type*)malloc((size_t)k_r * n_r * sizeof(BF16Type));
    reorder_B_bf16(B_orig, B_reo, k_r, n_r);

    AccumType *C = float32_zero(m, n_r);
    BF16Type *A_reo = (BF16Type*)malloc((size_t)(m + 8) * k_r * sizeof(BF16Type));

    struct timespec s, e;
    clock_gettime(CLOCK_MONOTONIC_RAW, &s);
    bf16gemm_k_dispatch(A, B_reo, C, m, k_r, n_r, A_reo);
    clock_gettime(CLOCK_MONOTONIC_RAW, &e);

    double t_us = get_time(&s, &e) * 1e6;
    double gflops = (2.0 * m * k_r * n_r) / get_time(&s, &e) * 1e-9;
    printf("time_used = %.3f us, FLOPS = %.3f GFLOPS\n", t_us, gflops);

    if (verify(A, B_orig, C, m, k_r, n_r)) {
        printf("successfully!\n");
    } else {
        printf("Validation failed!\n");
        free(A); free(B_orig); free(B_reo); free(C); free(A_reo);
        return 1;
    }

    free(A); free(B_orig); free(B_reo); free(C); free(A_reo);
    return 0;
}
