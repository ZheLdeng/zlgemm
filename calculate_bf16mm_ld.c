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

// bf16 kernel: M×K bf16 × K×N bf16 → M×N float32
void bf16gemm_k_ld(const BF16Type *A, const BF16Type *B_reordered,
                   AccumType *C, int m, int k, int n,
                   BF16Type *A_reorder, int lda, int ldb, int ldc);

// ── bf16 conversion helpers ──────────────────────────────────
static inline BF16Type float_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    // Round to nearest even (RNE)
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

// ── Timing ───────────────────────────────────────────────────
static double get_time(struct timespec *start, struct timespec *end) {
    return end->tv_sec - start->tv_sec +
        (end->tv_nsec - start->tv_nsec) * 1e-9;
}

// ── Matrix generation ────────────────────────────────────────
static BF16Type* bf16_generate(int rows, int cols) {
    BF16Type* m = (BF16Type*)malloc(rows * cols * sizeof(BF16Type));
    assert(m);
    for (int i = 0; i < rows * cols; i++) {
        // Generate values in [-2.0, 2.0) range, good for bf16 precision
        float val = ((float)rand() / (float)RAND_MAX) * 4.0f - 2.0f;
        m[i] = float_to_bf16(val);
    }
    return m;
}

static AccumType* float32_zero(int rows, int cols) {
    AccumType* m = (AccumType*)calloc(rows * cols, sizeof(AccumType));
    assert(m);
    return m;
}

// ── BF16 reference GEMM: C += bf16(A) * bf16(B) in float32 ──
static void bf16_ref_gemm(const BF16Type* A, const BF16Type* B,
                          AccumType* C, int m, int k, int n,
                          int lda, int ldb, int ldc) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                float av = bf16_to_float(A[i * lda + l]);
                float bv = bf16_to_float(B[l * ldb + j]);
                sum += av * bv;
            }
            C[i * ldc + j] += sum;
        }
    }
}

static void bf16_ref_gemm_simple(const BF16Type* A, const BF16Type* B,
                                  AccumType* C, int m, int k, int n) {
    bf16_ref_gemm(A, B, C, m, k, n, k, n, n);
}

// ── B reorder for bf16: N-major, K/4 blocks, column-major 4×2 ──
// Packed format: for each N/8 block, for each K/4 block:
//   4 groups of (4×2 col-major) = 4 cols × 2 col-pairs = 8 cols total
//   Each 4×2 group: 8 bf16 = 16 bytes
//   Total per K/4 block: 64 bytes
static void reorder_B_bf16(const BF16Type* B, BF16Type* B_reo, int K, int N) {
    assert(K % 4 == 0);
    assert(N % 8 == 0);
    int idx = 0;
    for (int cb = 0; cb < N/8; ++cb) {          // N blocks of 8
        for (int rb = 0; rb < K/4; ++rb) {      // K blocks of 4
            int row_base = rb * 4;
            int col_base = cb * 8;
            // 4 col-pairs: (0,1), (2,3), (4,5), (6,7)
            for (int cp = 0; cp < 4; ++cp) {
                int c0 = col_base + cp * 2;
                int c1 = c0 + 1;
                // Column-major within 4×2: col0 rows 0..3, then col1 rows 0..3
                for (int i = 0; i < 4; ++i)
                    B_reo[idx++] = B[(row_base + i) * N + c0];
                for (int i = 0; i < 4; ++i)
                    B_reo[idx++] = B[(row_base + i) * N + c1];
            }
        }
    }
}

// ── A reorder for bf16: M/8 × K/4 blocks, row-major 2×4 ──
// Format: for each M/8 block, for each K/4 block:
//   4 row-pairs × (2×4 row-major) = 8 rows × 4 cols total
//   Each 2×4 group: 8 bf16 = 16 bytes (row0[0..3], row1[0..3])
//   Total per block: 64 bytes
static void reorder_A_bf16(const BF16Type* A, BF16Type* A_reo,
                            int M, int K, int lda) {
    assert(M % 8 == 0);
    assert(K % 4 == 0);
    int idx = 0;
    for (int rb = 0; rb < M/8; ++rb) {          // M blocks of 8
        for (int cb = 0; cb < K/4; ++cb) {      // K blocks of 4
            int row_base = rb * 8;
            int col_base = cb * 4;
            // 4 row-pairs: (0,1), (2,3), (4,5), (6,7)
            for (int rp = 0; rp < 4; ++rp) {
                int r0 = row_base + rp * 2;
                int r1 = r0 + 1;
                // Row-major within 2×4: row0[0..3], row1[0..3]
                for (int j = 0; j < 4; ++j)
                    A_reo[idx++] = A[r0 * lda + col_base + j];
                for (int j = 0; j < 4; ++j)
                    A_reo[idx++] = A[r1 * lda + col_base + j];
            }
        }
    }
}

// ── Verification ─────────────────────────────────────────────
static int verify(const BF16Type* A, const BF16Type* B_orig,
                  const AccumType* C_result, const AccumType* C_init,
                  int m, int k, int n, int lda, int ldb, int ldc) {
    AccumType* ref = (AccumType*)calloc(m * ldc, sizeof(AccumType));
    // Copy initial C values
    for (int i = 0; i < m; i++)
        memcpy(ref + i * ldc, C_init + i * n, n * sizeof(AccumType));
    bf16_ref_gemm(A, B_orig, ref, m, k, n, lda, ldb, ldc);

    int ok = 1;
    float max_err = 0.0f;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float err = fabsf(C_result[i * ldc + j] - ref[i * ldc + j]);
            if (err > max_err) max_err = err;
            if (err > MATRIX_EPSILON) {
                printf("  Mismatch at (%d,%d): got=%f expect=%f err=%f\n",
                       i, j, (double)C_result[i * ldc + j],
                       (double)ref[i * ldc + j], (double)err);
                ok = 0;
                if (ok == 0) goto done;  // stop after first mismatch
            }
        }
    }
    if (ok) printf("  Max error: %e\n", (double)max_err);
done:
    free(ref);
    return ok;
}

// ── Test: Identity (lda=k, ldb=k, ldc=n) ─────────────────────
static int test_identity(int m, int k, int n) {
    srand(42);
    BF16Type* A = bf16_generate(m, k);
    BF16Type* B_orig = bf16_generate(k, n);
    BF16Type* B_packed = (BF16Type*)malloc(k * n * sizeof(BF16Type));
    reorder_B_bf16(B_orig, B_packed, k, n);

    AccumType* C_init = float32_zero(m, n);
    AccumType* C = (AccumType*)malloc(m * n * sizeof(AccumType));
    memcpy(C, C_init, m * n * sizeof(AccumType));
    BF16Type* A_reorder = (BF16Type*)malloc(m * k * sizeof(BF16Type));

    // lda, ldb, ldc are in element counts
    bf16gemm_k_ld(A, B_packed, C, m, k, n, A_reorder, k, k, n);

    int ok = verify(A, B_orig, C, C_init, m, k, n, k, n, n);

    // Verify A_reorder
    BF16Type* A_expected = (BF16Type*)malloc(m * k * sizeof(BF16Type));
    reorder_A_bf16(A, A_expected, m, k, k);
    if (memcmp(A_reorder, A_expected, m * k * sizeof(BF16Type)) != 0) {
        printf("  A_reorder mismatch!\n");
        ok = 0;
    }

    free(A); free(B_orig); free(B_packed); free(C_init); free(C);
    free(A_reorder); free(A_expected);
    return ok;
}

// ── Test: Sub-matrix A (lda > k) ─────────────────────────────
static int test_submatrix_A(int m, int k, int n, int lda, int k_offset) {
    srand(123);
    BF16Type* A_big = bf16_generate(m, lda);
    BF16Type* A_sub = A_big + k_offset;

    BF16Type* B_orig = bf16_generate(k, n);
    BF16Type* B_packed = (BF16Type*)malloc(k * n * sizeof(BF16Type));
    reorder_B_bf16(B_orig, B_packed, k, n);

    AccumType* C_init = float32_zero(m, n);
    AccumType* C = (AccumType*)malloc(m * n * sizeof(AccumType));
    memcpy(C, C_init, m * n * sizeof(AccumType));
    BF16Type* A_reorder = (BF16Type*)malloc(m * k * sizeof(BF16Type));

    bf16gemm_k_ld(A_sub, B_packed, C, m, k, n, A_reorder, lda, k, n);

    int ok = verify(A_sub, B_orig, C, C_init, m, k, n, lda, n, n);

    BF16Type* A_expected = (BF16Type*)malloc(m * k * sizeof(BF16Type));
    reorder_A_bf16(A_sub, A_expected, m, k, lda);
    if (memcmp(A_reorder, A_expected, m * k * sizeof(BF16Type)) != 0) {
        printf("  A_reorder mismatch!\n");
        ok = 0;
    }

    free(A_big); free(B_orig); free(B_packed); free(C_init); free(C);
    free(A_reorder); free(A_expected);
    return ok;
}

// ── Test: Sub-matrix B (ldb > k) ─────────────────────────────
static int test_submatrix_B(int m, int k, int n, int ldb, int k_offset) {
    assert(k_offset % 4 == 0);
    int k_offset_blocks = k_offset / 4;

    srand(456);
    BF16Type* A = bf16_generate(m, k);
    BF16Type* B_big_orig = bf16_generate(ldb, n);
    BF16Type* B_big_packed = (BF16Type*)malloc(ldb * n * sizeof(BF16Type));
    reorder_B_bf16(B_big_orig, B_big_packed, ldb, n);

    BF16Type* B_sub = B_big_packed + k_offset_blocks * 32; // each K/4 block = 32 bf16
    BF16Type* B_sub_orig = B_big_orig + k_offset * n;

    AccumType* C_init = float32_zero(m, n);
    AccumType* C = (AccumType*)malloc(m * n * sizeof(AccumType));
    memcpy(C, C_init, m * n * sizeof(AccumType));
    BF16Type* A_reorder = (BF16Type*)malloc(m * k * sizeof(BF16Type));

    bf16gemm_k_ld(A, B_sub, C, m, k, n, A_reorder, k, ldb, n);

    int ok = verify(A, B_sub_orig, C, C_init, m, k, n, k, n, n);

    free(A); free(B_big_orig); free(B_big_packed); free(C_init); free(C);
    free(A_reorder);
    return ok;
}

// ── Test: Sub-matrix C (ldc > n) ─────────────────────────────
static int test_submatrix_C(int m, int k, int n, int ldc, int n_offset) {
    srand(789);
    BF16Type* A = bf16_generate(m, k);
    BF16Type* B_orig = bf16_generate(k, n);
    BF16Type* B_packed = (BF16Type*)malloc(k * n * sizeof(BF16Type));
    reorder_B_bf16(B_orig, B_packed, k, n);

    AccumType* C_big_init = float32_zero(m, ldc);
    AccumType* C_big = (AccumType*)malloc(m * ldc * sizeof(AccumType));
    memcpy(C_big, C_big_init, m * ldc * sizeof(AccumType));
    AccumType* C_sub = C_big + n_offset;
    BF16Type* A_reorder = (BF16Type*)malloc(m * k * sizeof(BF16Type));

    bf16gemm_k_ld(A, B_packed, C_sub, m, k, n, A_reorder, k, k, ldc);

    AccumType* C_init_zero = float32_zero(m, n);
    int ok = verify(A, B_orig, C_sub, C_init_zero, m, k, n, k, n, ldc);

    // Check no corruption outside sub-matrix
    for (int i = 0; i < m && ok; i++) {
        for (int j = 0; j < n_offset && j < ldc; j++) {
            if (C_big[i * ldc + j] != 0.0f) {
                printf("  C corruption at (%d,%d): got=%f\n", i, j,
                       (double)C_big[i * ldc + j]);
                ok = 0;
            }
        }
    }
    for (int i = 0; i < m && ok; i++) {
        for (int j = n_offset + n; j < ldc; j++) {
            if (C_big[i * ldc + j] != 0.0f) {
                printf("  C corruption at (%d,%d): got=%f\n", i, j,
                       (double)C_big[i * ldc + j]);
                ok = 0;
            }
        }
    }

    free(A); free(B_orig); free(B_packed); free(C_big_init); free(C_big);
    free(C_init_zero); free(A_reorder);
    return ok;
}

// ── Test: Combined sub-matrices ──────────────────────────────
static int test_combined(int m, int k, int n, int lda, int ldb, int ldc,
                          int k_a_off, int k_b_off, int n_c_off) {
    assert(k_a_off % 4 == 0);
    assert(k_b_off % 4 == 0);
    int k_b_blocks = k_b_off / 4;

    srand(999);
    BF16Type* A_big = bf16_generate(m, lda);
    BF16Type* A_sub = A_big + k_a_off;

    BF16Type* B_big_orig = bf16_generate(ldb, n);
    BF16Type* B_big_packed = (BF16Type*)malloc(ldb * n * sizeof(BF16Type));
    reorder_B_bf16(B_big_orig, B_big_packed, ldb, n);
    BF16Type* B_sub = B_big_packed + k_b_blocks * 32;
    BF16Type* B_sub_orig = B_big_orig + k_b_off * n;

    AccumType* C_big_init = float32_zero(m, ldc);
    AccumType* C_big = (AccumType*)malloc(m * ldc * sizeof(AccumType));
    memcpy(C_big, C_big_init, m * ldc * sizeof(AccumType));
    AccumType* C_sub = C_big + n_c_off;

    BF16Type* A_reorder = (BF16Type*)malloc(m * k * sizeof(BF16Type));

    bf16gemm_k_ld(A_sub, B_sub, C_sub, m, k, n, A_reorder, lda, ldb, ldc);

    AccumType* C_init_zero = float32_zero(m, n);
    int ok = verify(A_sub, B_sub_orig, C_sub, C_init_zero, m, k, n, lda, n, ldc);

    free(A_big); free(B_big_orig); free(B_big_packed);
    free(C_big_init); free(C_big); free(C_init_zero);
    free(A_reorder);
    return ok;
}

// ── Main ─────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    int m, k, n;
    int do_sweep = 0;

    if (argc == 4) {
        m = atoi(argv[1]), k = atoi(argv[2]), n = atoi(argv[3]);
        do_sweep = 0;
    } else if (argc == 2 && strcmp(argv[1], "sweep") == 0) {
        do_sweep = 1;
    } else {
        m = 16, k = 16, n = 8;
    }

    int failed = 0, total = 0;

    if (do_sweep) {
        printf("=== BF16 GEMM Correctness Sweep ===\n");

        // K must be multiple of 8 for kernel
        int M_sizes[] = {8, 16, 24, 32, 48, 64};
        int K_sizes[] = {8, 16, 24, 32, 48, 64, 80};
        int N_sizes[] = {8, 16, 24, 32};

        // Phase 1: Identity tests
        printf("\n--- Phase 1: Identity ---\n");
        for (int mi = 0; mi < 6; mi++) {
            for (int ki = 0; ki < 7; ki++) {
                for (int ni = 0; ni < 4; ni++) {
                    total++;
                    if (!test_identity(M_sizes[mi], K_sizes[ki], N_sizes[ni])) {
                        printf("FAIL identity: M=%d K=%d N=%d\n",
                               M_sizes[mi], K_sizes[ki], N_sizes[ni]);
                        failed++;
                    }
                }
            }
        }

        // Phase 2: Sub-matrix A
        printf("\n--- Phase 2: Sub-matrix A ---\n");
        for (int mi = 0; mi < 4; mi++) {
            for (int ki = 0; ki < 4; ki++) {
                for (int ni = 0; ni < 2; ni++) {
                    int mm = M_sizes[mi], kk = K_sizes[ki], nn = N_sizes[ni];
                    int big_lda = kk * 2;
                    for (int off = 0; off < kk/4; off++) {
                        total++;
                        if (!test_submatrix_A(mm, kk, nn, big_lda, off * 4)) {
                            printf("FAIL sub-A: M=%d K=%d N=%d lda=%d off=%d\n",
                                   mm, kk, nn, big_lda, off * 4);
                            failed++;
                        }
                    }
                }
            }
        }

        // Phase 3: Sub-matrix B
        printf("\n--- Phase 3: Sub-matrix B ---\n");
        for (int mi = 0; mi < 4; mi++) {
            for (int ki = 0; ki < 4; ki++) {
                for (int ni = 0; ni < 2; ni++) {
                    int mm = M_sizes[mi], kk = K_sizes[ki], nn = N_sizes[ni];
                    int big_ldb = kk * 2;
                    for (int off = 0; off < kk/4; off++) {
                        total++;
                        if (!test_submatrix_B(mm, kk, nn, big_ldb, off * 4)) {
                            printf("FAIL sub-B: M=%d K=%d N=%d ldb=%d off=%d\n",
                                   mm, kk, nn, big_ldb, off * 4);
                            failed++;
                        }
                    }
                }
            }
        }

        // Phase 4: Sub-matrix C
        printf("\n--- Phase 4: Sub-matrix C ---\n");
        for (int mi = 0; mi < 4; mi++) {
            for (int ki = 0; ki < 3; ki++) {
                for (int ni = 0; ni < 2; ni++) {
                    int mm = M_sizes[mi], kk = K_sizes[ki], nn = N_sizes[ni];
                    int big_ldc = nn + 8;
                    total++;
                    if (!test_submatrix_C(mm, kk, nn, big_ldc, 4)) {
                        printf("FAIL sub-C: M=%d K=%d N=%d ldc=%d\n",
                               mm, kk, nn, big_ldc);
                        failed++;
                    }
                }
            }
        }

        // Phase 5: Combined
        printf("\n--- Phase 5: Combined ---\n");
        for (int mi = 0; mi < 3; mi++) {
            for (int ki = 0; ki < 2; ki++) {
                for (int ni = 0; ni < 2; ni++) {
                    int mm = M_sizes[mi], kk = K_sizes[ki], nn = N_sizes[ni];
                    int big_lda = kk + 16, big_ldb = kk + 16, big_ldc = nn + 16;
                    total++;
                    if (!test_combined(mm, kk, nn, big_lda, big_ldb, big_ldc,
                                       4, 4, 4)) {
                        printf("FAIL combined: M=%d K=%d N=%d lda=%d ldb=%d ldc=%d\n",
                               mm, kk, nn, big_lda, big_ldb, big_ldc);
                        failed++;
                    }
                }
            }
        }

        printf("\n=== Results: %d/%d passed ===\n", total - failed, total);
        return failed > 0 ? 1 : 0;
    }

    // ── Single-size test ──
    printf("BF16 GEMM: m=%d, k=%d, n=%d\n", m, k, n);

    srand(time(NULL));
    BF16Type* A = bf16_generate(m, k);
    BF16Type* B_orig = bf16_generate(k, n);
    BF16Type* B_reordered = (BF16Type*)malloc(k * n * sizeof(BF16Type));
    reorder_B_bf16(B_orig, B_reordered, k, n);

    AccumType* C = float32_zero(m, n);
    BF16Type* A_reorder = (BF16Type*)malloc(m * k * sizeof(BF16Type));

    AccumType* C_orig = (AccumType*)malloc(m * n * sizeof(AccumType));
    memcpy(C_orig, C, m * n * sizeof(AccumType));

    struct timespec start, end;
    int looptime = 1;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    for (int i = 0; i < looptime; i++)
        bf16gemm_k_ld(A, B_reordered, C, m, k, n, A_reorder, k, k, n);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    double time_used = get_time(&start, &end);
    double flops = 2.0 * m * n * k * (double)looptime / time_used * 1e-9;
    printf("time_used = %.3f us, FLOPS = %.3f GFLOPS\n",
           time_used * 1e6 / looptime, flops);

    // Verify
    AccumType* C_ref = (AccumType*)malloc(m * n * sizeof(AccumType));
    memcpy(C_ref, C_orig, m * n * sizeof(AccumType));
    bf16_ref_gemm_simple(A, B_orig, C_ref, m, k, n);

    int ok = 1;
    float max_err = 0.0f;
    for (int i = 0; i < m && ok; i++) {
        for (int j = 0; j < n && ok; j++) {
            float err = fabsf(C[i * n + j] - C_ref[i * n + j]);
            if (err > max_err) max_err = err;
            if (err > MATRIX_EPSILON) {
                printf("Mismatch at (%d,%d): got=%f expect=%f err=%f\n",
                       i, j, (double)C[i * n + j], (double)C_ref[i * n + j],
                       (double)err);
                ok = 0;
            }
        }
    }
    if (ok) {
        printf("successfully! (max error: %e)\n", (double)max_err);
    } else {
        printf("Validation failed!\n");
    }

    free(A); free(B_orig); free(B_reordered); free(C); free(C_orig);
    free(C_ref); free(A_reorder);
    return ok ? 0 : 1;
}
