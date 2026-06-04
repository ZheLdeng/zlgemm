// test_correctness.c
// Correctness test for bf16 and i8 tail-dispatch GEMM kernels.
// Tests unified dispatch: 8-row full blocks (main kernel) + tail (ld4/ld2/ld1).
//
// Build:
//   cc -o test_correctness test_correctness.c \
//      i8gemm_k.S bf16gemm_k.S \
//      -march=armv8.6-a+bf16+i8mm -O2 -Wall -lm
//
// Usage:
//   ./test_correctness [bf16|i8|all]   (default: all)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>

// ═══════════════════════════════════════════════════════════════════════
// Common helpers
// ═══════════════════════════════════════════════════════════════════════
static int is_power_of_two(int n) { return n > 0 && (n & (n - 1)) == 0; }

// ═══════════════════════════════════════════════════════════════════════
// BF16 precision
// ═══════════════════════════════════════════════════════════════════════
typedef float    BF16_Accum;
typedef uint16_t BF16_Type;

#define BF16_EPSILON 0.5f

static inline BF16_Type float_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    uint32_t rounding = ((u >> 16) & 1) + 0x7FFF;
    u += rounding;
    return (BF16_Type)(u >> 16);
}

static inline float bf16_to_float(BF16_Type b) {
    uint32_t u = ((uint32_t)b) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static void bf16_ref_gemm(const BF16_Type* A, const BF16_Type* B,
                           BF16_Accum* C, int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++)
                sum += bf16_to_float(A[i * k + l]) *
                       bf16_to_float(B[l * n + j]);
            C[i * n + j] += sum;
        }
}

void bf16gemm_k_ld (const BF16_Type *A, const BF16_Type *B_reo,
                    BF16_Accum *C, int m, int k, int n,
                    BF16_Type *A_reorder, int lda, int ldb, int ldc);
void bf16gemm_k_ld1(const BF16_Type *A, const BF16_Type *B_reo,
                    BF16_Accum *C, int m, int k, int n,
                    BF16_Type *A_reorder, int lda, int ldb, int ldc);
void bf16gemm_k_ld2(const BF16_Type *A, const BF16_Type *B_reo,
                    BF16_Accum *C, int m, int k, int n,
                    BF16_Type *A_reorder, int lda, int ldb, int ldc);
void bf16gemm_k_ld4(const BF16_Type *A, const BF16_Type *B_reo,
                    BF16_Accum *C, int m, int k, int n,
                    BF16_Type *A_reorder, int lda, int ldb, int ldc);

// bf16-store variants: C is bf16* instead of float*
void bf16gemm_k_ld_b (const BF16_Type *A, const BF16_Type *B_reo,
                      BF16_Type *C, int m, int k, int n,
                      BF16_Type *A_reorder, int lda, int ldb, int ldc);
void bf16gemm_k_ld1_b(const BF16_Type *A, const BF16_Type *B_reo,
                      BF16_Type *C, int m, int k, int n,
                      BF16_Type *A_reorder, int lda, int ldb, int ldc);
void bf16gemm_k_ld2_b(const BF16_Type *A, const BF16_Type *B_reo,
                      BF16_Type *C, int m, int k, int n,
                      BF16_Type *A_reorder, int lda, int ldb, int ldc);
void bf16gemm_k_ld4_b(const BF16_Type *A, const BF16_Type *B_reo,
                      BF16_Type *C, int m, int k, int n,
                      BF16_Type *A_reorder, int lda, int ldb, int ldc);

static void bf16_reorder_B(const BF16_Type* B, BF16_Type* B_reo,
                            int K, int N) {
    assert(K % 4 == 0 && N % 8 == 0);
    int idx = 0;
    for (int cb = 0; cb < N/8; ++cb)
        for (int rb = 0; rb < K/4; ++rb) {
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

static void bf16_dispatch(const BF16_Type *A, const BF16_Type *B_reo,
                           BF16_Accum *C, int m, int k, int n,
                           BF16_Type *A_reorder) {
    int lda = k, ldb = k, ldc = n;
    int processed = 0;

    int m_full = (m / 8) * 8;
    if (m_full > 0) {
        bf16gemm_k_ld(A, B_reo, C, m_full, k, n,
                      A_reorder, lda, ldb, ldc);
        processed = m_full;
    }

    int m_rem = m - processed;
    if (m_rem == 0) return;

    const BF16_Type *At = A      + (uint64_t)processed * k;
    BF16_Accum      *Ct = C     + (uint64_t)processed * n;
    BF16_Type *A_reo_t  = A_reorder + (uint64_t)processed * k;

    if (m_rem >= 4) {
        bf16gemm_k_ld4(At, B_reo, Ct, 4, k, n, A_reo_t, lda, ldb, ldc);
        processed += 4; m_rem -= 4;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * n;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 2) {
        bf16gemm_k_ld2(At, B_reo, Ct, 2, k, n, A_reo_t, lda, ldb, ldc);
        processed += 2; m_rem -= 2;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * n;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 1) {
        bf16gemm_k_ld1(At, B_reo, Ct, 1, k, n, A_reo_t, lda, ldb, ldc);
    }
}

static int bf16_verify(const BF16_Type* A, const BF16_Type* B_orig,
                        const BF16_Accum* C_result, int m, int k, int n) {
    BF16_Accum* ref = (BF16_Accum*)calloc(m * n, sizeof(BF16_Accum));
    bf16_ref_gemm(A, B_orig, ref, m, k, n);

    int ok = 1;
    float max_err = 0.0f;
    for (int i = 0; i < m && ok; i++)
        for (int j = 0; j < n && ok; j++) {
            float err = fabsf(C_result[i * n + j] - ref[i * n + j]);
            if (err > max_err) max_err = err;
            if (err > BF16_EPSILON) {
                printf("  BF16 MISMATCH at (%d,%d): got=%f expect=%f err=%e\n",
                       i, j, (double)C_result[i * n + j],
                       (double)ref[i * n + j], (double)err);
                ok = 0;
            }
        }
    if (ok) printf("  BF16 OK, max_err=%e\n", (double)max_err);
    free(ref);
    return ok;
}

static int bf16_test_single(int m, int k, int n) {
    int k_r = ((k + 7) / 8) * 8; if (k_r < 8)  k_r = 8;
    int n_r = ((n + 7) / 8) * 8; if (n_r < 8)  n_r = 8;

    srand(42);
    BF16_Type *A      = (BF16_Type*)malloc((size_t)m * k_r * sizeof(BF16_Type));
    BF16_Type *B_orig = (BF16_Type*)malloc((size_t)k_r * n_r * sizeof(BF16_Type));
    BF16_Type *B_reo  = (BF16_Type*)malloc((size_t)k_r * n_r * sizeof(BF16_Type));
    BF16_Accum *C     = (BF16_Accum*)calloc((size_t)m * n_r, sizeof(BF16_Accum));
    BF16_Type *A_reo  = (BF16_Type*)malloc((size_t)(m + 8) * k_r * sizeof(BF16_Type));

    for (int i = 0; i < m * k_r; i++)
        A[i] = float_to_bf16(((float)rand() / (float)RAND_MAX) * 4.0f - 2.0f);
    for (int i = 0; i < k_r * n_r; i++)
        B_orig[i] = float_to_bf16(((float)rand() / (float)RAND_MAX) * 4.0f - 2.0f);

    bf16_reorder_B(B_orig, B_reo, k_r, n_r);
    bf16_dispatch(A, B_reo, C, m, k_r, n_r, A_reo);
    int ok = bf16_verify(A, B_orig, C, m, k_r, n_r);

    free(A); free(B_orig); free(B_reo); free(C); free(A_reo);
    return ok;
}

static int bf16_sweep(void) {
    printf("\n=== BF16 Tail Correctness Sweep ===\n");
    int M_sizes[] = {1,2,3,4,5,6,7, 8,9,10,11,12,13,14,15,
                     16,17,18,19,20,21,22,23,
                     24,31,32,33,39,40,41,47,48,49,55,56,57,63,64,65};
    int K_sizes[] = {8, 16, 24, 32, 48, 64, 80, 128};
    int N_sizes[] = {8, 16, 24, 32, 64};
    int failed = 0, total = 0;
    for (int mi = 0; mi < (int)(sizeof(M_sizes)/sizeof(M_sizes[0])); mi++)
        for (int ki = 0; ki < (int)(sizeof(K_sizes)/sizeof(K_sizes[0])); ki++)
            for (int ni = 0; ni < (int)(sizeof(N_sizes)/sizeof(N_sizes[0])); ni++) {
                total++;
                int m = M_sizes[mi], k = K_sizes[ki], n = N_sizes[ni];
                if (!bf16_test_single(m, k, n)) {
                    printf("BF16 FAIL: M=%d K=%d N=%d\n", m, k, n);
                    failed++;
                }
            }
    printf("BF16: %d/%d passed\n\n", total - failed, total);
    return failed;
}

// ═══════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════
// BF16 bf16-store (_b suffix): C output is bf16 instead of float32
// ═══════════════════════════════════════════════════════════════════════

static void bf16_b_dispatch(const BF16_Type *A, const BF16_Type *B_reo,
                             BF16_Type *C, int m, int k, int n,
                             BF16_Type *A_reorder) {
    int lda = k, ldb = k, ldc = n;
    int processed = 0;
    int m_full = (m / 8) * 8;
    if (m_full > 0) {
        bf16gemm_k_ld_b(A, B_reo, C, m_full, k, n,
                        A_reorder, lda, ldb, ldc);
        processed = m_full;
    }
    int m_rem = m - processed;
    if (m_rem == 0) return;
    const BF16_Type *At = A      + (uint64_t)processed * k;
    BF16_Type       *Ct = C     + (uint64_t)processed * n;
    BF16_Type *A_reo_t  = A_reorder + (uint64_t)processed * k;
    if (m_rem >= 4) {
        bf16gemm_k_ld4_b(At, B_reo, Ct, 4, k, n, A_reo_t, lda, ldb, ldc);
        processed += 4; m_rem -= 4;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * n;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 2) {
        bf16gemm_k_ld2_b(At, B_reo, Ct, 2, k, n, A_reo_t, lda, ldb, ldc);
        processed += 2; m_rem -= 2;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * n;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 1)
        bf16gemm_k_ld1_b(At, B_reo, Ct, 1, k, n, A_reo_t, lda, ldb, ldc);
}

static int bf16_b_verify(const BF16_Type* A, const BF16_Type* B_orig,
                          const BF16_Type* C_result, int m, int k, int n) {
    BF16_Accum* ref32 = (BF16_Accum*)calloc(m * n, sizeof(BF16_Accum));
    bf16_ref_gemm(A, B_orig, ref32, m, k, n);
    int ok = 1;
    float max_err = 0.0f;
    for (int i = 0; i < m && ok; i++)
        for (int j = 0; j < n && ok; j++) {
            float got  = bf16_to_float(C_result[i * n + j]);
            float expect = ref32[i * n + j];
            float err = fabsf(got - expect);
            if (err > max_err) max_err = err;
            if (err > BF16_EPSILON) {
                printf("  BF16_B MISMATCH at (%d,%d): got=%f expect=%f err=%e\n",
                       i, j, (double)got, (double)expect, (double)err);
                ok = 0;
            }
        }
    if (ok) printf("  BF16_B OK, max_err=%e\n", (double)max_err);
    free(ref32);
    return ok;
}

static int bf16_b_test_single(int m, int k, int n) {
    int k_r = ((k + 7) / 8) * 8; if (k_r < 8)  k_r = 8;
    int n_r = ((n + 7) / 8) * 8; if (n_r < 8)  n_r = 8;
    srand(42);
    BF16_Type *A      = (BF16_Type*)malloc((size_t)m * k_r * sizeof(BF16_Type));
    BF16_Type *B_orig = (BF16_Type*)malloc((size_t)k_r * n_r * sizeof(BF16_Type));
    BF16_Type *B_reo  = (BF16_Type*)malloc((size_t)k_r * n_r * sizeof(BF16_Type));
    BF16_Type *C      = (BF16_Type*)calloc((size_t)m * n_r, sizeof(BF16_Type));
    BF16_Type *A_reo  = (BF16_Type*)malloc((size_t)(m + 8) * k_r * sizeof(BF16_Type));
    for (int i = 0; i < m * k_r; i++)
        A[i] = float_to_bf16(((float)rand() / (float)RAND_MAX) * 4.0f - 2.0f);
    for (int i = 0; i < k_r * n_r; i++)
        B_orig[i] = float_to_bf16(((float)rand() / (float)RAND_MAX) * 4.0f - 2.0f);
    bf16_reorder_B(B_orig, B_reo, k_r, n_r);
    bf16_b_dispatch(A, B_reo, C, m, k_r, n_r, A_reo);
    int ok = bf16_b_verify(A, B_orig, C, m, k_r, n_r);
    free(A); free(B_orig); free(B_reo); free(C); free(A_reo);
    return ok;
}

static int bf16_b_sweep(void) {
    printf("\n=== BF16 Store-bf16 Correctness Sweep ===\n");
    int M_sizes[] = {1,2,3,4,5,6,7, 8,9,10,11,12,13,14,15,
                     16,17,18,19,20,21,22,23,
                     24,31,32,33,39,40,41,47,48,49,55,56,57,63,64,65};
    int K_sizes[] = {8, 16, 24, 32, 48, 64, 80, 128};
    int N_sizes[] = {8, 16, 24, 32, 64};
    int failed = 0, total = 0;
    for (int mi = 0; mi < (int)(sizeof(M_sizes)/sizeof(M_sizes[0])); mi++)
        for (int ki = 0; ki < (int)(sizeof(K_sizes)/sizeof(K_sizes[0])); ki++)
            for (int ni = 0; ni < (int)(sizeof(N_sizes)/sizeof(N_sizes[0])); ni++) {
                total++;
                int m = M_sizes[mi], k = K_sizes[ki], n = N_sizes[ni];
                if (!bf16_b_test_single(m, k, n)) {
                    printf("BF16_B FAIL: M=%d K=%d N=%d\n", m, k, n);
                    failed++;
                }
            }
    printf("BF16_B: %d/%d passed\n\n", total - failed, total);
    return failed;
}

// ═══════════════════════════════════════════════════════════════════════
// BF16 bias fp32-output (_bias_f suffix) correctness tests
//
// C[i][j] = sum_k(A[i][k] * B[k][j]) + bias[j]
// Accumulators zero-init (bfmmla fp32), bias added in fp32 domain on store.
// ═══════════════════════════════════════════════════════════════════════

// BF16 bias kernel declarations (from bf16gemm_k_bias.S)
void bf16gemm_k_ld_bias_f (const BF16_Type *A, const BF16_Type *B_reo,
                            BF16_Accum *C, int m, int k, int n,
                            BF16_Type *A_reorder, int lda, int ldb, int ldc,
                            const float *bias);
void bf16gemm_k_ld1_bias_f(const BF16_Type *A, const BF16_Type *B_reo,
                            BF16_Accum *C, int m, int k, int n,
                            BF16_Type *A_reorder, int lda, int ldb, int ldc,
                            const float *bias);
void bf16gemm_k_ld2_bias_f(const BF16_Type *A, const BF16_Type *B_reo,
                            BF16_Accum *C, int m, int k, int n,
                            BF16_Type *A_reorder, int lda, int ldb, int ldc,
                            const float *bias);
void bf16gemm_k_ld4_bias_f(const BF16_Type *A, const BF16_Type *B_reo,
                            BF16_Accum *C, int m, int k, int n,
                            BF16_Type *A_reorder, int lda, int ldb, int ldc,
                            const float *bias);

static void bf16_bias_f_dispatch(const BF16_Type *A, const BF16_Type *B_reo,
                                  BF16_Accum *C, int m, int k, int n,
                                  BF16_Type *A_reorder, const float *bias) {
    int lda = k, ldb = k, ldc = n;
    int processed = 0;

    int m_full = (m / 8) * 8;
    if (m_full > 0) {
        bf16gemm_k_ld_bias_f(A, B_reo, C, m_full, k, n,
                             A_reorder, lda, ldb, ldc, bias);
        processed = m_full;
    }

    int m_rem = m - processed;
    if (m_rem == 0) return;

    const BF16_Type *At = A      + (uint64_t)processed * k;
    BF16_Accum      *Ct = C     + (uint64_t)processed * n;
    BF16_Type *A_reo_t  = A_reorder + (uint64_t)processed * k;

    if (m_rem >= 4) {
        bf16gemm_k_ld4_bias_f(At, B_reo, Ct, 4, k, n, A_reo_t,
                              lda, ldb, ldc, bias);
        processed += 4; m_rem -= 4;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * n;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 2) {
        bf16gemm_k_ld2_bias_f(At, B_reo, Ct, 2, k, n, A_reo_t,
                              lda, ldb, ldc, bias);
        processed += 2; m_rem -= 2;
        At = A + (uint64_t)processed * k;
        Ct = C + (uint64_t)processed * n;
        A_reo_t = A_reorder + (uint64_t)processed * k;
    }
    if (m_rem >= 1) {
        bf16gemm_k_ld1_bias_f(At, B_reo, Ct, 1, k, n, A_reo_t,
                              lda, ldb, ldc, bias);
    }
}

static int bf16_bias_f_verify(const BF16_Type* A, const BF16_Type* B_orig,
                               const BF16_Accum* C_result, int m, int k, int n,
                               const float* bias) {
    BF16_Accum* ref = (BF16_Accum*)calloc(m * n, sizeof(BF16_Accum));
    bf16_ref_gemm(A, B_orig, ref, m, k, n);

    int ok = 1;
    float max_err = 0.0f;
    for (int i = 0; i < m && ok; i++)
        for (int j = 0; j < n && ok; j++) {
            float got    = C_result[i * n + j];
            float expect = ref[i * n + j] + bias[j];
            float err = fabsf(got - expect);
            if (err > max_err) max_err = err;
            // Use same tolerance as bf16 fp32 output tests;
            // bfmmla accumulation order may differ slightly from scalar fp32.
            if (err > BF16_EPSILON) {
                printf("  BF16_BIAS_F MISMATCH at (%d,%d): got=%f expect=%f "
                       "err=%e (dot=%f bias=%f)\n",
                       i, j, (double)got, (double)expect, (double)err,
                       (double)ref[i * n + j], (double)bias[j]);
                ok = 0;
            }
        }
    if (ok) printf("  BF16_BIAS_F OK, max_err=%e\n", (double)max_err);
    free(ref);
    return ok;
}

static int bf16_bias_f_test_single(int m, int k, int n) {
    int k_r = ((k + 7) / 8) * 8; if (k_r < 8)  k_r = 8;
    int n_r = ((n + 7) / 8) * 8; if (n_r < 8)  n_r = 8;

    srand(42);
    BF16_Type *A      = (BF16_Type*)malloc((size_t)m * k_r * sizeof(BF16_Type));
    BF16_Type *B_orig = (BF16_Type*)malloc((size_t)k_r * n_r * sizeof(BF16_Type));
    BF16_Type *B_reo  = (BF16_Type*)malloc((size_t)k_r * n_r * sizeof(BF16_Type));
    BF16_Accum *C     = (BF16_Accum*)calloc((size_t)m * n_r, sizeof(BF16_Accum));
    BF16_Type *A_reo  = (BF16_Type*)malloc((size_t)(m + 8) * k_r * sizeof(BF16_Type));
    float     *bias   = (float*)malloc((size_t)n_r * sizeof(float));

    for (int i = 0; i < m * k_r; i++)
        A[i] = float_to_bf16(((float)rand() / (float)RAND_MAX) * 4.0f - 2.0f);
    for (int i = 0; i < k_r * n_r; i++)
        B_orig[i] = float_to_bf16(((float)rand() / (float)RAND_MAX) * 4.0f - 2.0f);
    for (int i = 0; i < n_r; i++)
        bias[i] = (float)((float)rand() / (float)RAND_MAX * 20.0f - 10.0f);

    bf16_reorder_B(B_orig, B_reo, k_r, n_r);
    bf16_bias_f_dispatch(A, B_reo, C, m, k_r, n_r, A_reo, bias);
    int ok = bf16_bias_f_verify(A, B_orig, C, m, k_r, n_r, bias);

    free(A); free(B_orig); free(B_reo); free(C); free(A_reo); free(bias);
    return ok;
}

static int bf16_bias_f_sweep(void) {
    printf("\n=== BF16 Bias-fp32 Correctness Sweep ===\n");
    int M_sizes[] = {1,2,3,4,5,6,7, 8,9,10,11,12,13,14,15,
                     16,17,18,19,20,21,22,23,
                     24,31,32,33,39,40,41,47,48,49,55,56,57,63,64,65};
    int K_sizes[] = {8, 16, 24, 32, 48, 64, 80, 128};
    int N_sizes[] = {8, 16, 24, 32, 64};
    int failed = 0, total = 0;
    for (int mi = 0; mi < (int)(sizeof(M_sizes)/sizeof(M_sizes[0])); mi++)
        for (int ki = 0; ki < (int)(sizeof(K_sizes)/sizeof(K_sizes[0])); ki++)
            for (int ni = 0; ni < (int)(sizeof(N_sizes)/sizeof(N_sizes[0])); ni++) {
                total++;
                int m = M_sizes[mi], k = K_sizes[ki], n = N_sizes[ni];
                if (!bf16_bias_f_test_single(m, k, n)) {
                    printf("BF16_BIAS_F FAIL: M=%d K=%d N=%d\n", m, k, n);
                    failed++;
                }
            }
    printf("BF16_BIAS_F: %d/%d passed\n\n", total - failed, total);
    return failed;
}
// I8 precision
// ═══════════════════════════════════════════════════════════════════════
typedef int32_t I8_Accum;
typedef int8_t  I8_Type;
typedef float    f32_t;

void i8gemm_k_ld (const I8_Type *A, const I8_Type *B_reo,
                  I8_Accum *C, int m, int k, int n,
                  I8_Type *A_reorder, int lda, int ldb, int ldc);
void i8gemm_k_ld1(const I8_Type *A, const I8_Type *B_reo,
                  I8_Accum *C, int m, int k, int n,
                  I8_Type *A_reorder, int lda, int ldb, int ldc);
void i8gemm_k_ld2(const I8_Type *A, const I8_Type *B_reo,
                  I8_Accum *C, int m, int k, int n,
                  I8_Type *A_reorder, int lda, int ldb, int ldc);
void i8gemm_k_ld4(const I8_Type *A, const I8_Type *B_reo,
                  I8_Accum *C, int m, int k, int n,
                  I8_Type *A_reorder, int lda, int ldb, int ldc);

// fp32-output variants (_f suffix): C is float*, A/B still int8
void i8gemm_k_ld_f (const I8_Type *A, const I8_Type *B_reo,
                    f32_t *C, int m, int k, int n,
                    I8_Type *A_reorder, int lda, int ldb, int ldc);
void i8gemm_k_ld1_f(const I8_Type *A, const I8_Type *B_reo,
                    f32_t *C, int m, int k, int n,
                    I8_Type *A_reorder, int lda, int ldb, int ldc);
void i8gemm_k_ld2_f(const I8_Type *A, const I8_Type *B_reo,
                    f32_t *C, int m, int k, int n,
                    I8_Type *A_reorder, int lda, int ldb, int ldc);
void i8gemm_k_ld4_f(const I8_Type *A, const I8_Type *B_reo,
                    f32_t *C, int m, int k, int n,
                    I8_Type *A_reorder, int lda, int ldb, int ldc);

static void i8_reorder_B(const I8_Type* B, I8_Type* B_reo, int K, int N) {
    assert(K % 8 == 0 && N % 8 == 0);
    int idx = 0;
    for (int cb = 0; cb < N/8; ++cb)
        for (int rb = 0; rb < K/8; ++rb)
            for (int j = 0; j < 8; ++j)
                for (int i = 0; i < 8; ++i)
                    B_reo[idx++] = B[(rb*8 + i) * N + (cb*8 + j)];
}

static void i8_ref_gemm(const I8_Type* A, const I8_Type* B,
                         I8_Accum* C, int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            int32_t sum = 0;
            for (int l = 0; l < k; l++)
                sum += (int32_t)A[i * k + l] * (int32_t)B[l * n + j];
            C[i * n + j] = sum;
        }
}

static void i8_dispatch(const I8_Type *A, const I8_Type *B_reo,
                         I8_Accum *C, int m, int k, int n,
                         I8_Type *A_reorder) {
    int lda = k, ldb = k, ldc = n;
    int processed = 0;

    int m_full = (m / 8) * 8;
    if (m_full > 0) {
        i8gemm_k_ld(A, B_reo, C, m_full, k, n,
                    A_reorder, lda, ldb, ldc);
        processed = m_full;
    }

    int m_rem = m - processed;
    if (m_rem == 0) return;

    const I8_Type *At = A      + processed * k;
    I8_Accum      *Ct = C     + processed * n;
    I8_Type *A_reo_t  = A_reorder + processed * k;

    if (m_rem >= 4) {
        i8gemm_k_ld4(At, B_reo, Ct, 4, k, n, A_reo_t, lda, ldb, ldc);
        processed += 4; m_rem -= 4;
        At = A + processed * k; Ct = C + processed * n;
        A_reo_t = A_reorder + processed * k;
    }
    if (m_rem >= 2) {
        i8gemm_k_ld2(At, B_reo, Ct, 2, k, n, A_reo_t, lda, ldb, ldc);
        processed += 2; m_rem -= 2;
        At = A + processed * k; Ct = C + processed * n;
        A_reo_t = A_reorder + processed * k;
    }
    if (m_rem >= 1) {
        i8gemm_k_ld1(At, B_reo, Ct, 1, k, n, A_reo_t, lda, ldb, ldc);
    }
}

static int i8_verify(const I8_Type* A, const I8_Type* B_orig,
                     const I8_Accum* C_result, int m, int k, int n) {
    I8_Accum* ref = (I8_Accum*)calloc(m * n, sizeof(I8_Accum));
    i8_ref_gemm(A, B_orig, ref, m, k, n);

    int ok = 1;
    for (int i = 0; i < m && ok; i++)
        for (int j = 0; j < n && ok; j++)
            if (C_result[i * n + j] != ref[i * n + j]) {
                printf("  I8 MISMATCH at (%d,%d): got=%d expect=%d\n",
                       i, j, C_result[i * n + j], ref[i * n + j]);
                ok = 0;
            }
    if (ok) printf("  I8 OK\n");
    free(ref);
    return ok;
}

static int i8_test_single(int m, int k, int n) {
    int k_r = ((k + 15) / 16) * 16; if (k_r < 16) k_r = 16;
    int n_r = ((n + 7)  / 8)  * 8;  if (n_r < 8)  n_r = 8;

    srand(42);
    I8_Type *A      = (I8_Type*)malloc((size_t)m * k_r);
    I8_Type *B_orig = (I8_Type*)malloc((size_t)k_r * n_r);
    I8_Type *B_reo  = (I8_Type*)malloc((size_t)k_r * n_r);
    I8_Accum *C     = (I8_Accum*)calloc((size_t)m * n_r, sizeof(I8_Accum));
    I8_Type *A_reo  = (I8_Type*)malloc((size_t)(m + 8) * k_r);

    for (int i = 0; i < m * k_r; i++)
        A[i] = (I8_Type)(rand() % 256 - 128);
    for (int i = 0; i < k_r * n_r; i++)
        B_orig[i] = (I8_Type)(rand() % 256);

    i8_reorder_B(B_orig, B_reo, k_r, n_r);
    i8_dispatch(A, B_reo, C, m, k_r, n_r, A_reo);
    int ok = i8_verify(A, B_orig, C, m, k_r, n_r);

    free(A); free(B_orig); free(B_reo); free(C); free(A_reo);
    return ok;
}

static int i8_sweep(void) {
    printf("\n=== I8 Tail Correctness Sweep ===\n");
    int M_sizes[] = {1,2,3,4,5,6,7, 8,9,10,11,12,13,14,15,
                     16,17,18,19,20,21,22,23,
                     24,31,32,33,39,40,41,47,48,49,55,56,57,63,64,65};
    int K_sizes[] = {16, 32, 48, 64, 80, 128};
    int N_sizes[] = {8, 16, 24, 32, 64};
    int failed = 0, total = 0;
    for (int mi = 0; mi < (int)(sizeof(M_sizes)/sizeof(M_sizes[0])); mi++)
        for (int ki = 0; ki < (int)(sizeof(K_sizes)/sizeof(K_sizes[0])); ki++)
            for (int ni = 0; ni < (int)(sizeof(N_sizes)/sizeof(N_sizes[0])); ni++) {
                total++;
                int m = M_sizes[mi], k = K_sizes[ki], n = N_sizes[ni];
                if (!i8_test_single(m, k, n)) {
                    printf("I8 FAIL: M=%d K=%d N=%d\n", m, k, n);
                    failed++;
                }
            }
    printf("I8: %d/%d passed\n\n", total - failed, total);
    return failed;
}

// ═══════════════════════════════════════════════════════════════════════
// I8 fp32-output (_f suffix) correctness tests
// ═══════════════════════════════════════════════════════════════════════

static void i8_f_dispatch(const I8_Type *A, const I8_Type *B_reo,
                           f32_t *C, int m, int k, int n,
                           I8_Type *A_reorder) {
    int lda = k, ldb = k, ldc = n;
    int processed = 0;

    int m_full = (m / 8) * 8;
    if (m_full > 0) {
        i8gemm_k_ld_f(A, B_reo, C, m_full, k, n,
                      A_reorder, lda, ldb, ldc);
        processed = m_full;
    }

    int m_rem = m - processed;
    if (m_rem == 0) return;

    const I8_Type *At = A      + processed * k;
    f32_t         *Ct = C     + processed * n;
    I8_Type *A_reo_t  = A_reorder + processed * k;

    if (m_rem >= 4) {
        i8gemm_k_ld4_f(At, B_reo, Ct, 4, k, n, A_reo_t, lda, ldb, ldc);
        processed += 4; m_rem -= 4;
        At = A + processed * k; Ct = C + processed * n;
        A_reo_t = A_reorder + processed * k;
    }
    if (m_rem >= 2) {
        i8gemm_k_ld2_f(At, B_reo, Ct, 2, k, n, A_reo_t, lda, ldb, ldc);
        processed += 2; m_rem -= 2;
        At = A + processed * k; Ct = C + processed * n;
        A_reo_t = A_reorder + processed * k;
    }
    if (m_rem >= 1) {
        i8gemm_k_ld1_f(At, B_reo, Ct, 1, k, n, A_reo_t, lda, ldb, ldc);
    }
}

static int i8_f_verify(const I8_Type* A, const I8_Type* B_orig,
                        const f32_t* C_result, int m, int k, int n) {
    I8_Accum* ref_i32 = (I8_Accum*)calloc(m * n, sizeof(I8_Accum));
    i8_ref_gemm(A, B_orig, ref_i32, m, k, n);

    int ok = 1;
    for (int i = 0; i < m && ok; i++)
        for (int j = 0; j < n && ok; j++) {
            float got    = C_result[i * n + j];
            float expect = (float)ref_i32[i * n + j];
            // int32→fp32 conversion via scvtf is exact for values up to 2^24,
            // and i8 GEMM values typically fit well within that range.
            if (got != expect) {
                printf("  I8_F MISMATCH at (%d,%d): got=%f expect=%f\n",
                       i, j, (double)got, (double)expect);
                ok = 0;
            }
        }
    if (ok) printf("  I8_F OK\n");
    free(ref_i32);
    return ok;
}

static int i8_f_test_single(int m, int k, int n) {
    int k_r = ((k + 15) / 16) * 16; if (k_r < 16) k_r = 16;
    int n_r = ((n + 7)  / 8)  * 8;  if (n_r < 8)  n_r = 8;

    srand(42);
    I8_Type *A      = (I8_Type*)malloc((size_t)m * k_r);
    I8_Type *B_orig = (I8_Type*)malloc((size_t)k_r * n_r);
    I8_Type *B_reo  = (I8_Type*)malloc((size_t)k_r * n_r);
    f32_t    *C     = (f32_t*)calloc((size_t)m * n_r, sizeof(f32_t));
    I8_Type *A_reo  = (I8_Type*)malloc((size_t)(m + 8) * k_r);

    for (int i = 0; i < m * k_r; i++)
        A[i] = (I8_Type)(rand() % 256 - 128);
    for (int i = 0; i < k_r * n_r; i++)
        B_orig[i] = (I8_Type)(rand() % 256);

    i8_reorder_B(B_orig, B_reo, k_r, n_r);
    i8_f_dispatch(A, B_reo, C, m, k_r, n_r, A_reo);
    int ok = i8_f_verify(A, B_orig, C, m, k_r, n_r);

    free(A); free(B_orig); free(B_reo); free(C); free(A_reo);
    return ok;
}

static int i8_f_sweep(void) {
    printf("\n=== I8 Store-fp32 Correctness Sweep ===\n");
    int M_sizes[] = {1,2,3,4,5,6,7, 8,9,10,11,12,13,14,15,
                     16,17,18,19,20,21,22,23,
                     24,31,32,33,39,40,41,47,48,49,55,56,57,63,64,65};
    int K_sizes[] = {16, 32, 48, 64, 80, 128};
    int N_sizes[] = {8, 16, 24, 32, 64};
    int failed = 0, total = 0;
    for (int mi = 0; mi < (int)(sizeof(M_sizes)/sizeof(M_sizes[0])); mi++)
        for (int ki = 0; ki < (int)(sizeof(K_sizes)/sizeof(K_sizes[0])); ki++)
            for (int ni = 0; ni < (int)(sizeof(N_sizes)/sizeof(N_sizes[0])); ni++) {
                total++;
                int m = M_sizes[mi], k = K_sizes[ki], n = N_sizes[ni];
                if (!i8_f_test_single(m, k, n)) {
                    printf("I8_F FAIL: M=%d K=%d N=%d\n", m, k, n);
                    failed++;
                }
            }
    printf("I8_F: %d/%d passed\n\n", total - failed, total);
    return failed;
}

// ═══════════════════════════════════════════════════════════════════════
// I8 bias fp32-output (_bias_f suffix) correctness tests
//
// C[i][j] = sum_k(A[i][k] * B[k][j]) + bias[j]
// Accumulators zero-init, bias added in fp32 domain on store.
// ═══════════════════════════════════════════════════════════════════════

// Bias kernel declarations (from i8gemm_k_bias.S)
void i8gemm_k_ld_bias_f (const I8_Type *A, const I8_Type *B_reo,
                          f32_t *C, int m, int k, int n,
                          I8_Type *A_reorder, int lda, int ldb, int ldc,
                          const f32_t *bias);
void i8gemm_k_ld1_bias_f(const I8_Type *A, const I8_Type *B_reo,
                          f32_t *C, int m, int k, int n,
                          I8_Type *A_reorder, int lda, int ldb, int ldc,
                          const f32_t *bias);
void i8gemm_k_ld2_bias_f(const I8_Type *A, const I8_Type *B_reo,
                          f32_t *C, int m, int k, int n,
                          I8_Type *A_reorder, int lda, int ldb, int ldc,
                          const f32_t *bias);
void i8gemm_k_ld4_bias_f(const I8_Type *A, const I8_Type *B_reo,
                          f32_t *C, int m, int k, int n,
                          I8_Type *A_reorder, int lda, int ldb, int ldc,
                          const f32_t *bias);

static void i8_bias_f_dispatch(const I8_Type *A, const I8_Type *B_reo,
                                f32_t *C, int m, int k, int n,
                                I8_Type *A_reorder, const f32_t *bias) {
    int lda = k, ldb = k, ldc = n;
    int processed = 0;

    int m_full = (m / 8) * 8;
    if (m_full > 0) {
        i8gemm_k_ld_bias_f(A, B_reo, C, m_full, k, n,
                           A_reorder, lda, ldb, ldc, bias);
        processed = m_full;
    }

    int m_rem = m - processed;
    if (m_rem == 0) return;

    const I8_Type *At = A      + processed * k;
    f32_t         *Ct = C     + processed * n;
    I8_Type *A_reo_t  = A_reorder + processed * k;

    if (m_rem >= 4) {
        i8gemm_k_ld4_bias_f(At, B_reo, Ct, 4, k, n, A_reo_t,
                            lda, ldb, ldc, bias);
        processed += 4; m_rem -= 4;
        At = A + processed * k; Ct = C + processed * n;
        A_reo_t = A_reorder + processed * k;
    }
    if (m_rem >= 2) {
        i8gemm_k_ld2_bias_f(At, B_reo, Ct, 2, k, n, A_reo_t,
                            lda, ldb, ldc, bias);
        processed += 2; m_rem -= 2;
        At = A + processed * k; Ct = C + processed * n;
        A_reo_t = A_reorder + processed * k;
    }
    if (m_rem >= 1) {
        i8gemm_k_ld1_bias_f(At, B_reo, Ct, 1, k, n, A_reo_t,
                            lda, ldb, ldc, bias);
    }
}

static int i8_bias_f_verify(const I8_Type* A, const I8_Type* B_orig,
                              const f32_t* C_result, int m, int k, int n,
                              const f32_t* bias) {
    // Compute reference: int32 GEMM + bias → float
    I8_Accum* ref_i32 = (I8_Accum*)calloc(m * n, sizeof(I8_Accum));
    i8_ref_gemm(A, B_orig, ref_i32, m, k, n);

    int ok = 1;
    for (int i = 0; i < m && ok; i++)
        for (int j = 0; j < n && ok; j++) {
            float got    = C_result[i * n + j];
            float expect = (float)ref_i32[i * n + j] + bias[j];
            if (got != expect) {
                printf("  I8_BIAS_F MISMATCH at (%d,%d): got=%f expect=%f "
                       "(dot=%d bias=%f)\n",
                       i, j, (double)got, (double)expect,
                       ref_i32[i * n + j], (double)bias[j]);
                ok = 0;
            }
        }
    if (ok) printf("  I8_BIAS_F OK\n");
    free(ref_i32);
    return ok;
}

static int i8_bias_f_test_single(int m, int k, int n) {
    int k_r = ((k + 15) / 16) * 16; if (k_r < 16) k_r = 16;
    int n_r = ((n + 7)  / 8)  * 8;  if (n_r < 8)  n_r = 8;

    srand(42);
    I8_Type *A      = (I8_Type*)malloc((size_t)m * k_r);
    I8_Type *B_orig = (I8_Type*)malloc((size_t)k_r * n_r);
    I8_Type *B_reo  = (I8_Type*)malloc((size_t)k_r * n_r);
    f32_t    *C     = (f32_t*)calloc((size_t)m * n_r, sizeof(f32_t));
    I8_Type *A_reo  = (I8_Type*)malloc((size_t)(m + 8) * k_r);
    f32_t    *bias  = (f32_t*)malloc((size_t)n_r * sizeof(f32_t));

    // Generate random data (same pattern as existing i8 tests)
    for (int i = 0; i < m * k_r; i++)
        A[i] = (I8_Type)(rand() % 256 - 128);
    for (int i = 0; i < k_r * n_r; i++)
        B_orig[i] = (I8_Type)(rand() % 256);
    // Random bias values in a reasonable range
    for (int i = 0; i < n_r; i++)
        bias[i] = (f32_t)((float)rand() / (float)RAND_MAX * 20.0f - 10.0f);

    i8_reorder_B(B_orig, B_reo, k_r, n_r);
    i8_bias_f_dispatch(A, B_reo, C, m, k_r, n_r, A_reo, bias);
    int ok = i8_bias_f_verify(A, B_orig, C, m, k_r, n_r, bias);

    free(A); free(B_orig); free(B_reo); free(C); free(A_reo); free(bias);
    return ok;
}

static int i8_bias_f_sweep(void) {
    printf("\n=== I8 Bias-fp32 Correctness Sweep ===\n");
    int M_sizes[] = {1,2,3,4,5,6,7, 8,9,10,11,12,13,14,15,
                     16,17,18,19,20,21,22,23,
                     24,31,32,33,39,40,41,47,48,49,55,56,57,63,64,65};
    int K_sizes[] = {16, 32, 48, 64, 80, 128};
    int N_sizes[] = {8, 16, 24, 32, 64};
    int failed = 0, total = 0;
    for (int mi = 0; mi < (int)(sizeof(M_sizes)/sizeof(M_sizes[0])); mi++)
        for (int ki = 0; ki < (int)(sizeof(K_sizes)/sizeof(K_sizes[0])); ki++)
            for (int ni = 0; ni < (int)(sizeof(N_sizes)/sizeof(N_sizes[0])); ni++) {
                total++;
                int m = M_sizes[mi], k = K_sizes[ki], n = N_sizes[ni];
                if (!i8_bias_f_test_single(m, k, n)) {
                    printf("I8_BIAS_F FAIL: M=%d K=%d N=%d\n", m, k, n);
                    failed++;
                }
            }
    printf("I8_BIAS_F: %d/%d passed\n\n", total - failed, total);
    return failed;
}

// ═══════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════
int main(int argc, char *argv[]) {
    const char *mode = (argc >= 2) ? argv[1] : "all";

    if (strcmp(mode, "all") != 0 && strcmp(mode, "bf16") != 0 &&
        strcmp(mode, "bf16bias") != 0 &&
        strcmp(mode, "i8")   != 0 && strcmp(mode, "i8f")  != 0 &&
        strcmp(mode, "i8bias") != 0) {
        fprintf(stderr, "Usage: %s [bf16|bf16bias|i8|i8f|i8bias|all]\n", argv[0]);
        return 1;
    }

    int total_failed = 0;

    if (strcmp(mode, "all") == 0 || strcmp(mode, "i8") == 0) {
        total_failed += i8_sweep();
    }
    if (strcmp(mode, "all") == 0 || strcmp(mode, "i8f") == 0) {
        total_failed += i8_f_sweep();
    }
    if (strcmp(mode, "all") == 0 || strcmp(mode, "i8bias") == 0) {
        total_failed += i8_bias_f_sweep();
    }
    if (strcmp(mode, "all") == 0 || strcmp(mode, "bf16") == 0) {
        total_failed += bf16_sweep();
        total_failed += bf16_b_sweep();
    }
    if (strcmp(mode, "all") == 0 || strcmp(mode, "bf16bias") == 0) {
        total_failed += bf16_bias_f_sweep();
    }

    if (total_failed > 0) {
        printf("*** %d FAILURES ***\n", total_failed);
        return 1;
    }
    printf("All tests passed.\n");
    return 0;
}
