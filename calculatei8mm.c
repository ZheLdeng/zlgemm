#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <assert.h>
//#include "sme_matrix.h"
typedef int32_t MatrixType;
typedef int8_t DFMatrixType;
typedef u_int8_t BMatrixType;
#define MATRIX_EPSILON 1e-5

void i8gemm_k(const DFMatrixType *A, const BMatrixType *B, const MatrixType *C, int m, int k, int n);

static double get_time(struct timespec *start,
    struct timespec *end)
{
    return end->tv_sec - start->tv_sec +
        (end->tv_nsec - start->tv_nsec) * 1e-9;
}
// 生成随机矩阵
DFMatrixType* dfmatrix_generate(int rows, int cols) {
    DFMatrixType* matrix = (DFMatrixType*)malloc(rows * cols * sizeof(DFMatrixType));
    if (!matrix) exit(EXIT_FAILURE);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            //matrix[i * cols + j] = (MatrixType)rand() / RAND_MAX;
            // matrix[i * cols + j] = (j + 1) * (i + 1);
            matrix[i * cols + j] = i;
            // matrix[i * cols + j] = 0;
        }
    }
    return matrix;
}
BMatrixType* bmatrix_generate(int rows, int cols) {
    BMatrixType* matrix = (BMatrixType*)malloc(rows * cols * sizeof(BMatrixType));
    if (!matrix) exit(EXIT_FAILURE);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            //matrix[i * cols + j] = (MatrixType)rand() / RAND_MAX;
            // matrix[i * cols + j] = (j + 1) * (i + 1);
            matrix[i * cols + j] = j;
            // matrix[i * cols + j] = 0;
        }
    }
    return matrix;
}
DFMatrixType* romatrix_generate(int rows, int cols) {
    DFMatrixType* matrix = (DFMatrixType*)malloc(rows * cols * sizeof(DFMatrixType));
    if (!matrix) exit(EXIT_FAILURE);
    
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            // matrix[i * cols + j] = (j + 1) * (i + 1);
            matrix[i * rows + j] = j + 1;
            //matrix[i * rows + j] = 2 + i;
            //matrix[i * rows + j] = (MatrixType)rand() / RAND_MAX;
        }
    }
    return matrix;
}
// 生成全0矩阵
MatrixType* zero_generate(int rows, int cols) {
    MatrixType* matrix = (MatrixType*)malloc(rows * cols * sizeof(MatrixType));
    if (!matrix) exit(EXIT_FAILURE);
    
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = 0;
        // if ( i > 64 * 8) {
        //     matrix[i] = 2;
        // }
    }
    return matrix;
}
// 矩阵乘法计算 C = A*B + C
void matrix_calculate(const DFMatrixType* A, const BMatrixType* B,
                      MatrixType* C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            MatrixType sum = 0;
            for (int l = 0; l < k; l++) {
                sum += (MatrixType)A[i * k + l] * (MatrixType)B[l * n + j];
            }
            C[i * n + j] += sum;
        }
    }
}

// A: 原始 M×K 行主序，A_reordered: 输出 buffer
// 8×8 为一个 block，block 内行主序，block 之间行主序
void reorder_A_1x8(const int8_t* A, int8_t* A_reordered, int M, int K) {
    assert(M % 8 == 0);  // 假设 M 能被 8 整除
    assert(K % 8 == 0);  // 假设 K 能被 8 整除

    int idx = 0;
    int num_row_blocks = M / 8;
    int num_col_blocks = K / 8;

    // block 之间行主序：先遍历行块，再遍历列块
    for (int rb = 0; rb < num_row_blocks; ++rb) {
        for (int cb = 0; cb < num_col_blocks; ++cb) {
            int row_base = rb * 8;
            int col_base = cb * 8;
            // block 内行主序
            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j) {
                    A_reordered[idx++] = A[(row_base + i) * K + (col_base + j)];
                }
            }
        }
    }
}

// B: 原始 K×N 行主序，B_reordered: 输出 buffer
// 8×8 为一个 block，block 内列主序，block 之间列主序
void reorder_B_8x1(const BMatrixType* B, BMatrixType* B_reordered, int K, int N) {
    assert(K % 8 == 0);  // 假设 K 能被 8 整除
    assert(N % 8 == 0);  // 假设 N 能被 8 整除
    
    int idx = 0;
    int num_row_blocks = K / 8;
    int num_col_blocks = N / 8;

    // block 之间列主序：先遍历列块，再遍历行块
    for (int cb = 0; cb < num_col_blocks; ++cb) {
        for (int rb = 0; rb < num_row_blocks; ++rb) {
            int row_base = rb * 8;
            int col_base = cb * 8;
            // block 内列主序
            for (int j = 0; j < 8; ++j) {
                for (int i = 0; i < 8; ++i) {
                    B_reordered[idx++] = B[(row_base + i) * N + (col_base + j)];
                }
            }
        }
    }
}

// C: 2×2 block 格式还原为普通行主序
// 输入 C_blocked: 2×2 为一个 block，block 内行主序，block 之间行主序
// 输出 C_normal: 普通 M×N 行主序
void reorder_C_2x2(const MatrixType* C_blocked, MatrixType* C_normal, int M, int N) {
    assert(M % 2 == 0);  // 假设 M 能被 2 整除
    assert(N % 2 == 0);  // 假设 N 能被 2 整除

    int idx = 0;
    int num_row_blocks = M / 2;
    int num_col_blocks = N / 2;

    // block 之间行主序：先遍历行块，再遍历列块
    for (int rb = 0; rb < num_row_blocks; ++rb) {
        for (int cb = 0; cb < num_col_blocks; ++cb) {
            int row_base = rb * 2;
            int col_base = cb * 2;
            // block 内行主序，还原到普通行主序
            C_normal[(row_base + 0) * N + (col_base + 0)] = C_blocked[idx++];
            C_normal[(row_base + 0) * N + (col_base + 1)] = C_blocked[idx++];
            C_normal[(row_base + 1) * N + (col_base + 0)] = C_blocked[idx++];
            C_normal[(row_base + 1) * N + (col_base + 1)] = C_blocked[idx++];
        }
    }
}


// 结果验证函数
int result_check(const DFMatrixType* A, const BMatrixType* B, const MatrixType* C_orig,
                 MatrixType* C_result, int m, int k, int n) {
    int valid = 1;
    MatrixType* reference = (MatrixType*)malloc(m*n * sizeof(MatrixType));
    memcpy(reference, C_orig, m*n * sizeof(MatrixType));

    // 计算参考值
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            MatrixType sum = 0;
            for (int l = 0; l < k; l++) {
                sum += (MatrixType)A[i * k + l] * (MatrixType)B[l * n + j];
            }
            reference[i * n + j] += sum;
            
            // 直接比较计算结果
            if (abs(C_result[i * n + j] - reference[i * n + j]) > MATRIX_EPSILON) {
                printf("Mismatch at (%d, %d): C=%.4d vs Ref=%.4d\n",
                      i, j, C_result[i * n + j], reference[i * n + j]);
                valid = 0;
                goto cleanup;
            }
        }
    }

cleanup:
    free(reference);
    return valid;
}

// 打印矩阵（调试用）
void matrix_print(const MatrixType* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

void row_dfmatrix_print(const DFMatrixType* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2hhd ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}
void row_bmatrix_print(const BMatrixType* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2hhd ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}
void col_dfmatrix_print(const DFMatrixType* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2hhd ", matrix[j * rows + i]);
        }
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    int m ,n, k;
    double time_used, flops;
    struct timespec start, end; 
    // 参数验证
    if (argc != 4) {
        m = 16, k = 16, n = 8;
    } else {
        int dims[3];
        for (int i = 0; i < 3; i++) {
            char* endptr;
            dims[i] = strtol(argv[i+1], &endptr, 10);
        }
        m = dims[0], k = dims[1], n = dims[2];
    }
    printf("%d , %d, %d \n ", m ,k ,n);
    // 转换并验证参数
    // 生成矩阵
    DFMatrixType* A = dfmatrix_generate(m, k);
    BMatrixType* B = bmatrix_generate(k, n);
    MatrixType* C = zero_generate(m, n);  // 带初始值的C矩阵
    int8_t* A_reordered = malloc(m * k);
    BMatrixType* B_reordered = malloc(k * n);
    reorder_A_1x8(A, A_reordered, m, k);
    reorder_B_8x1(B, B_reordered, k, n);
    // 保留原始C的副本用于验证
    MatrixType* C_orig = (MatrixType*)malloc(m*n * sizeof(MatrixType));
    MatrixType* C_reordered = (MatrixType*)malloc(m*n * sizeof(MatrixType));
    memcpy(C_orig, C, m * n * sizeof(MatrixType));
    MatrixType* C_orig2 = (MatrixType*)malloc(m*n * sizeof(MatrixType));
    memcpy(C_orig2, C, m * n * sizeof(MatrixType));
    printf("start \n");
    // 执行矩阵计算
    int looptime = 1;
    matrix_calculate(A, B, C_orig2, m, k, n);
    long long ret = 0;
    // for (int i = 0; i < 10; i++) {
    //     i8gemm_k(A_reordered, B_reordered, C, m, k, n);
    // }
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    for (int i = 0; i < looptime; i++) {
        i8gemm_k(A_reordered, B_reordered, C, m, k, n);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    time_used = get_time(&start, &end);
    flops = 2 * m * n * k * (double)looptime / time_used * 1e-9;
    printf("size = %ld , looptime = %ld\n", 2 * m * n * k, looptime);
    printf("time_used = %lf ms, FLOPS = %lf GFLOPS\n", time_used * 1e6 / looptime, flops);
    reorder_C_2x2(C, C_reordered, m, n);
    
    // // 调试时可取消注释查看矩阵内容
    printf("\nMatrix A:\n");
    row_dfmatrix_print(A, m, k);
    printf("\nReorder A:\n");
    row_dfmatrix_print(A_reordered, m, k);
    printf("\nMatrix B:\n");
    row_bmatrix_print(B, k, n);
    printf("\nReorder B:\n");
    row_bmatrix_print(B_reordered, k, n);
    printf("\nCorrect Matrix C:\n");
    matrix_print(C_orig2, m, n); 
    printf("\nResult Matrix C:\n");
    matrix_print(C_reordered, m, n);
    if (result_check(A, B, C_orig, C_reordered, m, k, n)) {
        printf("successfully!\n");
    } else {    
        printf("Validation failed!\n");
    }
    // printf("end\n");
    // 释放内存
    free(A); free(B); free(C); free(C_orig); free(C_orig2);
    return EXIT_SUCCESS;
}