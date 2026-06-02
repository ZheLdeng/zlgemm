#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <assert.h>

typedef int32_t MatrixType;
typedef int8_t  DFMatrixType;
typedef int8_t  BMatrixType;
#define MATRIX_EPSILON 1e-5

// New ld kernel with lda, ldb, ldc
void i8gemm_k_ld(const DFMatrixType *A, const BMatrixType *B_reordered,
                 MatrixType *C, int m, int k, int n,
                 int8_t *A_reorder, int lda, int ldb, int ldc);

static double get_time(struct timespec *start, struct timespec *end) {
    return end->tv_sec - start->tv_sec +
        (end->tv_nsec - start->tv_nsec) * 1e-9;
}

static DFMatrixType* dfmatrix_generate(int rows, int cols) {
    DFMatrixType* matrix = (DFMatrixType*)malloc(rows * cols * sizeof(DFMatrixType));
    if (!matrix) exit(EXIT_FAILURE);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix[i * cols + j] = (DFMatrixType)(rand() % 256 - 128);
    return matrix;
}

static BMatrixType* bmatrix_generate(int rows, int cols) {
    BMatrixType* matrix = (BMatrixType*)malloc(rows * cols * sizeof(BMatrixType));
    if (!matrix) exit(EXIT_FAILURE);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix[i * cols + j] = (BMatrixType)(rand() % 256);
    return matrix;
}

static MatrixType* zero_generate(int rows, int cols) {
    MatrixType* matrix = (MatrixType*)calloc(rows * cols, sizeof(MatrixType));
    if (!matrix) exit(EXIT_FAILURE);
    return matrix;
}

// Reference GEMM: C += A * B (A: M×K, B: K×N, all row-major)
static void matrix_calculate(const DFMatrixType* A, const BMatrixType* B,
                              MatrixType* C, int m, int k, int n, int lda, int ldb, int ldc) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            MatrixType sum = 0;
            for (int l = 0; l < k; l++) {
                sum += (MatrixType)A[i * lda + l] * (MatrixType)B[l * ldb + j];
            }
            C[i * ldc + j] += sum;
        }
    }
}

// Simplified reference for contiguous case (lda=k, ldb=n)
static void matrix_calculate_simple(const DFMatrixType* A, const BMatrixType* B,
                                     MatrixType* C, int m, int k, int n) {
    matrix_calculate(A, B, C, m, k, n, k, n, n);
}

// B packing: N-major 8×8 blocks, block-internal column-major
// same as calculatei8mm.c's reorder_B_8x1
static void reorder_B_8x1(const BMatrixType* B, BMatrixType* B_reordered,
                           int K, int N) {
    assert(K % 8 == 0);
    assert(N % 8 == 0);
    int idx = 0;
    for (int cb = 0; cb < N/8; ++cb) {
        for (int rb = 0; rb < K/8; ++rb) {
            int row_base = rb * 8;
            int col_base = cb * 8;
            for (int j = 0; j < 8; ++j) {
                for (int i = 0; i < 8; ++i) {
                    B_reordered[idx++] = B[(row_base + i) * N + (col_base + j)];
                }
            }
        }
    }
}

// Reorder A to match kernel-internal format (for verification)
static void reorder_A_1x8(const int8_t* A, int8_t* A_reordered, int M, int K, int lda) {
    assert(M % 8 == 0);
    assert(K % 8 == 0);
    int idx = 0;
    for (int rb = 0; rb < M/8; ++rb) {
        for (int cb = 0; cb < K/8; ++cb) {
            int row_base = rb * 8;
            int col_base = cb * 8;
            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j) {
                    A_reordered[idx++] = A[(row_base + i) * lda + (col_base + j)];
                }
            }
        }
    }
}

// Verify kernel result against reference
static int verify(const DFMatrixType* A, const BMatrixType* B_orig,
                  const MatrixType* C_result, const MatrixType* C_init,
                  int m, int k, int n, int lda, int ldb, int ldc) {
    MatrixType* reference = (MatrixType*)calloc(m * ldc, sizeof(MatrixType));
    // Copy only the initialized columns from C_init (C_init may have stride != ldc)
    // C_init is always m×n row-major with stride n
    for (int i = 0; i < m; i++)
        memcpy(reference + i * ldc, C_init + i * n, n * sizeof(MatrixType));

    matrix_calculate(A, B_orig, reference, m, k, n, lda, ldb, ldc);

    int valid = 1;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (abs(C_result[i * ldc + j] - reference[i * ldc + j]) > MATRIX_EPSILON) {
                printf("  Mismatch at (%d,%d): got=%d expect=%d\n",
                       i, j, C_result[i * ldc + j], reference[i * ldc + j]);
                valid = 0;
                goto done;
            }
        }
    }
done:
    free(reference);
    return valid;
}

// ── Test: Identity case (lda=k, ldb=k, ldc=n) ──
static int test_identity(int m, int k, int n) {
    srand(42);
    DFMatrixType* A = dfmatrix_generate(m, k);
    BMatrixType* B_orig = bmatrix_generate(k, n);
    BMatrixType* B_packed = malloc(k * n);
    reorder_B_8x1(B_orig, B_packed, k, n);

    MatrixType* C_init = zero_generate(m, n);
    MatrixType* C = malloc(m * n * sizeof(MatrixType));
    memcpy(C, C_init, m * n * sizeof(MatrixType));
    int8_t* A_reorder = malloc(m * k);

    i8gemm_k_ld(A, B_packed, C, m, k, n, A_reorder, k, k, n);

    int ok = verify(A, B_orig, C, C_init, m, k, n, k, n, n);

    // Also verify A_reorder matches expected
    int8_t* A_expected = malloc(m * k);
    reorder_A_1x8(A, A_expected, m, k, k);
    int areo_ok = !memcmp(A_reorder, A_expected, m * k);
    if (!areo_ok) { printf("  A_reorder mismatch!\n"); ok = 0; }

    free(A); free(B_orig); free(B_packed); free(C_init); free(C);
    free(A_reorder); free(A_expected);
    return ok;
}

// ── Test: Sub-matrix A (lda > k) ──
static int test_submatrix_A(int m, int k, int n, int lda, int k_offset) {
    // Big A: m × lda, sub-matrix: m × k starting at column k_offset
    srand(123);
    DFMatrixType* A_big = dfmatrix_generate(m, lda);
    DFMatrixType* A_sub = A_big + k_offset;  // pointer into big A

    BMatrixType* B_orig = bmatrix_generate(k, n);
    BMatrixType* B_packed = malloc(k * n);
    reorder_B_8x1(B_orig, B_packed, k, n);

    MatrixType* C_init = zero_generate(m, n);
    MatrixType* C = malloc(m * n * sizeof(MatrixType));
    memcpy(C, C_init, m * n * sizeof(MatrixType));
    int8_t* A_reorder = malloc(m * k);

    i8gemm_k_ld(A_sub, B_packed, C, m, k, n, A_reorder, lda, k, n);

    // Reference: multiply A_sub (as m×k matrix with row stride lda) × B_orig
    int ok = verify(A_sub, B_orig, C, C_init, m, k, n, lda, n, n);

    // Verify A_reorder
    int8_t* A_expected = malloc(m * k);
    reorder_A_1x8(A_sub, A_expected, m, k, lda);
    int areo_ok = !memcmp(A_reorder, A_expected, m * k);
    if (!areo_ok) { printf("  A_reorder mismatch!\n"); ok = 0; }

    free(A_big); free(B_orig); free(B_packed); free(C_init); free(C);
    free(A_reorder); free(A_expected);
    return ok;
}

// ── Test: Sub-matrix B (ldb > k) ──
static int test_submatrix_B(int m, int k, int n, int ldb, int k_offset) {
    // Big B: ldb × n, packed. Sub-matrix: k × n starting at K block k_offset/8
    // k_offset must be a multiple of 8
    assert(k_offset % 8 == 0);
    int k_offset_blocks = k_offset / 8;

    srand(456);
    DFMatrixType* A = dfmatrix_generate(m, k);
    BMatrixType* B_big_orig = bmatrix_generate(ldb, n);
    BMatrixType* B_big_packed = malloc(ldb * n);
    reorder_B_8x1(B_big_orig, B_big_packed, ldb, n);

    // Pointer to sub-region in packed B: skip k_offset_blocks K-blocks
    BMatrixType* B_sub = B_big_packed + k_offset_blocks * 64;

    // The sub-matrix B in original form: rows k_offset .. k_offset+k-1
    BMatrixType* B_sub_orig = B_big_orig + k_offset * n;

    MatrixType* C_init = zero_generate(m, n);
    MatrixType* C = malloc(m * n * sizeof(MatrixType));
    memcpy(C, C_init, m * n * sizeof(MatrixType));
    int8_t* A_reorder = malloc(m * k);

    i8gemm_k_ld(A, B_sub, C, m, k, n, A_reorder, k, ldb, n);

    int ok = verify(A, B_sub_orig, C, C_init, m, k, n, k, n, n);

    free(A); free(B_big_orig); free(B_big_packed); free(C_init); free(C);
    free(A_reorder);
    return ok;
}

// ── Test: Sub-matrix C (ldc > n) ──
static int test_submatrix_C(int m, int k, int n, int ldc, int n_offset) {
    // Big C: m × ldc, sub-matrix: m × n starting at column n_offset
    srand(789);
    DFMatrixType* A = dfmatrix_generate(m, k);
    BMatrixType* B_orig = bmatrix_generate(k, n);
    BMatrixType* B_packed = malloc(k * n);
    reorder_B_8x1(B_orig, B_packed, k, n);

    MatrixType* C_big_init = zero_generate(m, ldc);
    MatrixType* C_big = malloc(m * ldc * sizeof(MatrixType));
    memcpy(C_big, C_big_init, m * ldc * sizeof(MatrixType));
    MatrixType* C_sub = C_big + n_offset;
    int8_t* A_reorder = malloc(m * k);

    i8gemm_k_ld(A, B_packed, C_sub, m, k, n, A_reorder, k, k, ldc);

    // Verify: C_sub should be A*B
    MatrixType* C_init_zero = zero_generate(m, n);
    int ok = verify(A, B_orig, C_sub, C_init_zero, m, k, n, k, n, ldc);

    // Also check that C_big outside the sub-matrix is untouched
    // (columns 0..n_offset-1 and n_offset+n..ldc-1 should be 0)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n_offset && j < ldc; j++) {
            if (C_big[i * ldc + j] != 0) {
                printf("  C corruption at (%d,%d): got=%d expect=0\n", i, j, C_big[i * ldc + j]);
                ok = 0; goto done;
            }
        }
    }
    for (int i = 0; i < m; i++) {
        for (int j = n_offset + n; j < ldc; j++) {
            if (C_big[i * ldc + j] != 0) {
                printf("  C corruption at (%d,%d): got=%d expect=0\n", i, j, C_big[i * ldc + j]);
                ok = 0; goto done;
            }
        }
    }
done:
    free(A); free(B_orig); free(B_packed); free(C_big_init); free(C_big);
    free(A_reorder);
    return ok;
}

// ── Test: Combined sub-matrices (lda > k, ldb > k, ldc > n) ──
static int test_combined(int m, int k, int n, int lda, int ldb, int ldc,
                          int k_a_offset, int k_b_offset, int n_c_offset) {
    assert(k_a_offset % 8 == 0);
    assert(k_b_offset % 8 == 0);
    int k_b_offset_blocks = k_b_offset / 8;

    srand(999);
    // Big A: m × lda
    DFMatrixType* A_big = dfmatrix_generate(m, lda);
    DFMatrixType* A_sub = A_big + k_a_offset;

    // Big B: ldb × n, packed
    BMatrixType* B_big_orig = bmatrix_generate(ldb, n);
    BMatrixType* B_big_packed = malloc(ldb * n);
    reorder_B_8x1(B_big_orig, B_big_packed, ldb, n);
    BMatrixType* B_sub = B_big_packed + k_b_offset_blocks * 64;
    BMatrixType* B_sub_orig = B_big_orig + k_b_offset * n;

    // Big C: m × ldc
    MatrixType* C_big_init = zero_generate(m, ldc);
    MatrixType* C_big = malloc(m * ldc * sizeof(MatrixType));
    memcpy(C_big, C_big_init, m * ldc * sizeof(MatrixType));
    MatrixType* C_sub = C_big + n_c_offset;

    int8_t* A_reorder = malloc(m * k);

    i8gemm_k_ld(A_sub, B_sub, C_sub, m, k, n, A_reorder, lda, ldb, ldc);

    MatrixType* C_init_zero = zero_generate(m, n);
    int ok = verify(A_sub, B_sub_orig, C_sub, C_init_zero, m, k, n, lda, n, ldc);

    free(A_big); free(B_big_orig); free(B_big_packed);
    free(C_big_init); free(C_big); free(C_init_zero);
    free(A_reorder);
    return ok;
}

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
        // ── Sweep: random sizes + sub-matrix scenarios ──
        printf("=== Correctness Sweep ===\n");

        // Phase 1: Identity tests (lda=k, ldb=k, ldc=n)
        int M_sizes[] = {8, 16, 24, 32, 40, 48, 56, 64};
        int K_sizes[] = {16, 32, 48, 64, 80};
        int N_sizes[] = {8, 16, 24, 32};

        for (int mi = 0; mi < 8; mi++) {
            for (int ki = 0; ki < 5; ki++) {
                for (int ni = 0; ni < 4; ni++) {
                    total++;
                    if (!test_identity(M_sizes[mi], K_sizes[ki], N_sizes[ni])) {
                        printf("FAIL identity: M=%d K=%d N=%d\n", M_sizes[mi], K_sizes[ki], N_sizes[ni]);
                        failed++;
                    }
                }
            }
        }

        // Phase 2: Sub-matrix A tests
        for (int mi = 0; mi < 4; mi++) {
            for (int ki = 0; ki < 3; ki++) {
                for (int ni = 0; ni < 2; ni++) {
                    int mm = M_sizes[mi], kk = K_sizes[ki], nn = N_sizes[ni];
                    int big_lda = kk * 2;  // lda = 2*k
                    for (int off = 0; off < kk/8; off++) {
                        total++;
                        if (!test_submatrix_A(mm, kk, nn, big_lda, off * 8)) {
                            printf("FAIL sub-A: M=%d K=%d N=%d lda=%d off=%d\n", mm, kk, nn, big_lda, off*8);
                            failed++;
                        }
                    }
                }
            }
        }

        // Phase 3: Sub-matrix B tests
        for (int mi = 0; mi < 4; mi++) {
            for (int ki = 0; ki < 3; ki++) {
                for (int ni = 0; ni < 2; ni++) {
                    int mm = M_sizes[mi], kk = K_sizes[ki], nn = N_sizes[ni];
                    int big_ldb = kk * 2;  // ldb = 2*k
                    for (int off = 0; off < kk/8; off++) {
                        total++;
                        if (!test_submatrix_B(mm, kk, nn, big_ldb, off * 8)) {
                            printf("FAIL sub-B: M=%d K=%d N=%d ldb=%d off=%d\n", mm, kk, nn, big_ldb, off*8);
                            failed++;
                        }
                    }
                }
            }
        }

        // Phase 4: Sub-matrix C tests
        for (int mi = 0; mi < 4; mi++) {
            for (int ki = 0; ki < 3; ki++) {
                for (int ni = 0; ni < 2; ni++) {
                    int mm = M_sizes[mi], kk = K_sizes[ki], nn = N_sizes[ni];
                    int big_ldc = (nn + 8);  // ldc = n + 8
                    total++;
                    if (!test_submatrix_C(mm, kk, nn, big_ldc, 4)) {
                        printf("FAIL sub-C: M=%d K=%d N=%d ldc=%d\n", mm, kk, nn, big_ldc);
                        failed++;
                    }
                }
            }
        }

        // Phase 5: Combined sub-matrix tests
        for (int mi = 0; mi < 3; mi++) {
            for (int ki = 0; ki < 2; ki++) {
                for (int ni = 0; ni < 2; ni++) {
                    int mm = M_sizes[mi], kk = K_sizes[ki], nn = N_sizes[ni];
                    int big_lda = kk + 16, big_ldb = kk + 16, big_ldc = nn + 16;
                    total++;
                    if (!test_combined(mm, kk, nn, big_lda, big_ldb, big_ldc, 8, 8, 8)) {
                        printf("FAIL combined: M=%d K=%d N=%d lda=%d ldb=%d ldc=%d\n",
                               mm, kk, nn, big_lda, big_ldb, big_ldc);
                        failed++;
                    }
                }
            }
        }

        printf("Results: %d/%d passed\n", total - failed, total);
        return failed > 0 ? 1 : 0;
    }

    // ── Single-size test (print details) ──
    printf("m=%d, k=%d, n=%d\n", m, k, n);

    srand(time(NULL));
    DFMatrixType* A = dfmatrix_generate(m, k);
    BMatrixType* B_orig = bmatrix_generate(k, n);
    BMatrixType* B_reordered = malloc(k * n);
    reorder_B_8x1(B_orig, B_reordered, k, n);

    MatrixType* C = zero_generate(m, n);
    int8_t* A_reorder = malloc(m * k);

    MatrixType* C_orig = malloc(m * n * sizeof(MatrixType));
    memcpy(C_orig, C, m * n * sizeof(MatrixType));

    struct timespec start, end;
    int looptime = 1;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    for (int i = 0; i < looptime; i++)
        i8gemm_k_ld(A, B_reordered, C, m, k, n, A_reorder, k, k, n);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    double time_used = get_time(&start, &end);
    double flops = 2.0 * m * n * k * (double)looptime / time_used * 1e-9;
    printf("time_used = %.3f us, FLOPS = %.3f GFLOPS\n", time_used * 1e6 / looptime, flops);

    // Verify
    MatrixType* C_ref = malloc(m * n * sizeof(MatrixType));
    memcpy(C_ref, C_orig, m * n * sizeof(MatrixType));
    matrix_calculate_simple(A, B_orig, C_ref, m, k, n);

    int ok = 1;
    for (int i = 0; i < m && ok; i++) {
        for (int j = 0; j < n && ok; j++) {
            if (abs(C[i * n + j] - C_ref[i * n + j]) > MATRIX_EPSILON) {
                printf("Mismatch at (%d,%d): got=%d expect=%d\n",
                       i, j, C[i * n + j], C_ref[i * n + j]);
                ok = 0;
            }
        }
    }
    if (ok) printf("successfully!\n");
    else printf("Validation failed!\n");

    free(A); free(B_orig); free(B_reordered); free(C); free(C_orig);
    free(C_ref); free(A_reorder);
    return ok ? 0 : 1;
}
