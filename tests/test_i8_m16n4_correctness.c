// test_i8_m16n4_correctness.c -- Compare experimental m16n4 dispatch with
// the default i8gemm NEON dispatch.

#include "i8gemm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int round_up(int x, int q) {
    return ((x + q - 1) / q) * q;
}

static int check_shape(int M, int K, int N, int nth) {
    const int K_r = round_up(K < 16 ? 16 : K, 16);
    const int N_r = round_up(N < 8 ? 8 : N, 8);
    i8_t *A = (i8_t *)calloc((size_t)M * (size_t)K_r, 1);
    i8_t *B = (i8_t *)calloc((size_t)K_r * (size_t)N_r, 1);
    i8_t *B_reo = (i8_t *)aligned_alloc(64, (size_t)K_r * (size_t)N_r);
    i32_t *C0 = (i32_t *)calloc((size_t)M * (size_t)N_r, sizeof(i32_t));
    i32_t *C1 = (i32_t *)calloc((size_t)M * (size_t)N_r, sizeof(i32_t));
    if (!A || !B || !B_reo || !C0 || !C1)
        return 2;

    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            A[(size_t)i * (size_t)K_r + (size_t)k] =
                (i8_t)(((i * 3 + k * 5) % 17) - 8);
    for (int k = 0; k < K; ++k)
        for (int j = 0; j < N; ++j)
            B[(size_t)k * (size_t)N_r + (size_t)j] =
                (i8_t)(((k * 7 + j * 11) % 13) - 6);

    i8_pack_B(B, B_reo, K_r, N_r);
    i8gemm_mt_dispatch(A, B_reo, C0, M, K_r, N_r, nth);
    i8gemm_mt_dispatch_m16n4(A, B_reo, C1, M, K_r, N_r, nth);

    int bad = 0;
    for (int i = 0; i < M && !bad; ++i) {
        for (int j = 0; j < N_r; ++j) {
            const i32_t a = C0[(size_t)i * (size_t)N_r + (size_t)j];
            const i32_t b = C1[(size_t)i * (size_t)N_r + (size_t)j];
            if (a != b) {
                fprintf(stderr,
                        "mismatch M=%d K=%d N=%d nth=%d at (%d,%d): %d vs %d\n",
                        M, K, N, nth, i, j, a, b);
                bad = 1;
                break;
            }
        }
    }

    free(A);
    free(B);
    free(B_reo);
    free(C0);
    free(C1);
    return bad;
}

int main(void) {
    const int shapes[][3] = {
        {16, 128, 16}, {16, 128, 128}, {17, 31, 13},
        {31, 128, 24}, {64, 512, 512}, {128, 256, 32},
    };
    for (int s = 0; s < (int)(sizeof(shapes) / sizeof(shapes[0])); ++s) {
        for (int nth = 1; nth <= 4; nth *= 2) {
            int rc = check_shape(shapes[s][0], shapes[s][1], shapes[s][2], nth);
            if (rc)
                return rc;
        }
    }
    puts("m16n4 correctness ok");
    return 0;
}
