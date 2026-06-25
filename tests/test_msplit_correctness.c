// Correctness test for the M-split / split-K lib vs a naive int32 reference.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "i8gemm_msplit.h"

static void naive(const int8_t *A, const int8_t *B, int32_t *C,
                  int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[i * K + k] * (int32_t)B[k * N + j];
            C[i * N + j] = acc;
        }
}

static int check_shape(int M, int K, int N, int nt, unsigned seed) {
    int8_t *A = malloc((size_t)M * K);
    int8_t *B = malloc((size_t)K * N);
    int32_t *Cr = malloc((size_t)M * N * sizeof(int32_t));
    int32_t *Cx = calloc((size_t)M * N, sizeof(int32_t));
    srand(seed);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (int8_t)((rand() % 255) - 127);
    for (size_t i = 0; i < (size_t)K * N; i++) B[i] = (int8_t)((rand() % 255) - 127);

    naive(A, B, Cr, M, K, N);
    i8gemm_msplit(A, B, Cx, M, K, N, nt);

    int bad = 0;
    for (size_t i = 0; i < (size_t)M * N && bad < 5; i++)
        if (Cr[i] != Cx[i]) {
            fprintf(stderr, "  MISMATCH %dx%dx%d t%d idx=%zu ref=%d got=%d\n",
                    M, K, N, nt, i, Cr[i], Cx[i]);
            bad++;
        }
    free(A); free(B); free(Cr); free(Cx);
    return bad == 0;
}

int main(void) {
    int Ms[] = {1, 2, 4, 7, 8, 12, 13, 16, 31, 32, 48, 64, 96, 128, 200, 256};
    int Ks[] = {16, 32, 64, 128, 256, 512, 1024};
    int Ns[] = {8, 16, 24, 64, 128, 256, 512, 1000};
    int Ts[] = {1, 2, 3, 4, 5, 7, 8, 13};
    int total = 0, ok = 0;
    unsigned seed = 12345;
    for (size_t a = 0; a < sizeof(Ms)/sizeof(int); a++)
      for (size_t b = 0; b < sizeof(Ks)/sizeof(int); b++)
        for (size_t c = 0; c < sizeof(Ns)/sizeof(int); c++)
          for (size_t d = 0; d < sizeof(Ts)/sizeof(int); d++) {
              total++;
              if (check_shape(Ms[a], Ks[b], Ns[c], Ts[d], seed++)) ok++;
          }
    printf("msplit correctness: %d/%d cases passed\n", ok, total);
    return ok == total ? 0 : 1;
}
