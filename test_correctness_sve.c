// test_correctness_sve.c -- correctness coverage for the SVE GEMM variant.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <arm_sve.h>

#include "bf16gemm.h"
#include "i8gemm.h"

#define BF16_EPSILON 0.5f
#define SENTINEL_F32 1234567.0f
#define SENTINEL_I32 0x5a5a1234
#define SENTINEL_BF16 0x7e55u

static bf16_t test_float_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    u += ((u >> 16) & 1u) + 0x7fffu;
    return (bf16_t)(u >> 16);
}

static float test_bf16_to_float(bf16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static float frand_range(float lo, float hi) {
    return lo + (hi - lo) * ((float)rand() / (float)RAND_MAX);
}

static int round_up_int(int x, int q) {
    return ((x + q - 1) / q) * q;
}

static int sve_n_tile(void) {
    return (int)(svcntb() / 16) * 8;
}

static int check_f32_sentinel(const float *buf, size_t used, size_t guard) {
    for (size_t i = 0; i < guard; i++)
        if (buf[used + i] != SENTINEL_F32)
            return 0;
    return 1;
}

static int check_i32_sentinel(const int32_t *buf, size_t used, size_t guard) {
    for (size_t i = 0; i < guard; i++)
        if (buf[used + i] != SENTINEL_I32)
            return 0;
    return 1;
}

static int check_bf16_sentinel(const bf16_t *buf, size_t used, size_t guard) {
    for (size_t i = 0; i < guard; i++)
        if (buf[used + i] != SENTINEL_BF16)
            return 0;
    return 1;
}

static void ref_bf16(const bf16_t *A, const bf16_t *B, const float *bias,
                     float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < K; k++)
                sum += test_bf16_to_float(A[(size_t)i * K + k]) *
                       test_bf16_to_float(B[(size_t)k * N + j]);
            C[(size_t)i * N + j] = sum;
        }
    }
}

static void ref_i8(const int8_t *A, const int8_t *B, int32_t *C,
                   int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++)
                sum += (int32_t)A[(size_t)i * K + k] *
                       (int32_t)B[(size_t)k * N + j];
            C[(size_t)i * N + j] = sum;
        }
    }
}

static int verify_bf16_dispatch_case(const bf16_t *A, const bf16_t *B,
                                     const float *bias, const float *ref,
                                     int M, int K, int N, int nth) {
    int K_r = round_up_int(K < 8 ? 8 : K, 8);
    int N_r = round_up_int(N < 8 ? 8 : N, sve_n_tile());
    const size_t pad_elems = (size_t)M * N_r;
    const size_t guard = 32;
    bf16_t *A_pad = (bf16_t *)calloc((size_t)M * K_r, sizeof(bf16_t));
    bf16_t *B_pad = (bf16_t *)calloc((size_t)K_r * N_r, sizeof(bf16_t));
    bf16_t *B_reo = (bf16_t *)malloc((size_t)K_r * N_r * sizeof(bf16_t));
    float *bias_pad = (float *)calloc((size_t)N_r, sizeof(float));
    float *C = (float *)calloc(pad_elems + guard, sizeof(float));
    float *Cbias = (float *)calloc(pad_elems + guard, sizeof(float));
    bf16_t *Cbf16 = (bf16_t *)calloc(pad_elems + guard, sizeof(bf16_t));
    if (!A_pad || !B_pad || !B_reo || !bias_pad || !C || !Cbias || !Cbf16)
        return 0;

    for (int i = 0; i < M; i++)
        memcpy(A_pad + (size_t)i * K_r, A + (size_t)i * K, (size_t)K * sizeof(bf16_t));
    for (int i = 0; i < K; i++)
        memcpy(B_pad + (size_t)i * N_r, B + (size_t)i * N, (size_t)N * sizeof(bf16_t));
    memcpy(bias_pad, bias, (size_t)N * sizeof(float));
    bf16_pack_B(B_pad, B_reo, K_r, N_r);
    for (size_t i = 0; i < guard; i++) {
        C[pad_elems + i] = SENTINEL_F32;
        Cbias[pad_elems + i] = SENTINEL_F32;
        Cbf16[pad_elems + i] = SENTINEL_BF16;
    }

    bf16gemm_mt_dispatch(A_pad, B_reo, C, M, K_r, N_r, nth);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            size_t got_idx = (size_t)i * N_r + j;
            size_t ref_idx = (size_t)i * N + j;
            if (fabsf(C[got_idx] - ref[ref_idx]) > BF16_EPSILON) {
                printf("BF16 dispatch mismatch M=%d K=%d N=%d nth=%d (%d,%d) got=%f ref=%f\n",
                       M, K, N, nth, i, j, C[got_idx], ref[ref_idx]);
                return 0;
            }
        }
    }
    if (!check_f32_sentinel(C, pad_elems, guard))
        return 0;

    bf16gemm_mt_dispatch_bias_f(A_pad, B_reo, Cbias, M, K_r, N_r, nth, bias_pad);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            size_t got_idx = (size_t)i * N_r + j;
            size_t ref_idx = (size_t)i * N + j;
            float expect = ref[ref_idx] + bias[j];
            if (fabsf(Cbias[got_idx] - expect) > BF16_EPSILON) {
                printf("BF16 dispatch bias mismatch M=%d K=%d N=%d nth=%d (%d,%d) got=%f ref=%f\n",
                       M, K, N, nth, i, j, Cbias[got_idx], expect);
                return 0;
            }
        }
    }
    if (!check_f32_sentinel(Cbias, pad_elems, guard))
        return 0;

    bf16gemm_mt_dispatch_nld_b(A_pad, B_reo, Cbf16, M, K_r, N_r, nth);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            size_t got_idx = (size_t)i * N_r + j;
            size_t ref_idx = (size_t)i * N + j;
            bf16_t expect = test_float_to_bf16(ref[ref_idx]);
            if (fabsf(test_bf16_to_float(Cbf16[got_idx]) - test_bf16_to_float(expect)) > BF16_EPSILON) {
                printf("BF16 dispatch nld_b mismatch M=%d K=%d N=%d nth=%d (%d,%d)\n",
                       M, K, N, nth, i, j);
                return 0;
            }
        }
    }
    if (!check_bf16_sentinel(Cbf16, pad_elems, guard))
        return 0;

    free(A_pad);
    free(B_pad);
    free(B_reo);
    free(bias_pad);
    free(C);
    free(Cbias);
    free(Cbf16);
    return 1;
}

static int verify_i8_dispatch_case(const int8_t *A, const int8_t *B,
                                   const float *bias, const int32_t *ref,
                                   int M, int K, int N, int nth) {
    int K_r = round_up_int(K < 16 ? 16 : K, 16);
    int N_r = round_up_int(N < 8 ? 8 : N, sve_n_tile());
    const size_t pad_elems = (size_t)M * N_r;
    const size_t guard = 32;
    int8_t *A_pad = (int8_t *)calloc((size_t)M * K_r, 1);
    int8_t *B_pad = (int8_t *)calloc((size_t)K_r * N_r, 1);
    int8_t *B_reo = (int8_t *)malloc((size_t)K_r * N_r);
    float *bias_pad = (float *)calloc((size_t)N_r, sizeof(float));
    int32_t *C = (int32_t *)calloc(pad_elems + guard, sizeof(int32_t));
    float *Cf = (float *)calloc(pad_elems + guard, sizeof(float));
    float *Cbias = (float *)calloc(pad_elems + guard, sizeof(float));
    if (!A_pad || !B_pad || !B_reo || !bias_pad || !C || !Cf || !Cbias)
        return 0;

    for (int i = 0; i < M; i++)
        memcpy(A_pad + (size_t)i * K_r, A + (size_t)i * K, (size_t)K);
    for (int i = 0; i < K; i++)
        memcpy(B_pad + (size_t)i * N_r, B + (size_t)i * N, (size_t)N);
    memcpy(bias_pad, bias, (size_t)N * sizeof(float));
    i8_pack_B(B_pad, B_reo, K_r, N_r);
    for (size_t i = 0; i < guard; i++) {
        C[pad_elems + i] = SENTINEL_I32;
        Cf[pad_elems + i] = SENTINEL_F32;
        Cbias[pad_elems + i] = SENTINEL_F32;
    }

    i8gemm_mt_dispatch(A_pad, B_reo, C, M, K_r, N_r, nth);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            size_t got_idx = (size_t)i * N_r + j;
            size_t ref_idx = (size_t)i * N + j;
            if (C[got_idx] != ref[ref_idx]) {
                printf("I8 dispatch mismatch M=%d K=%d N=%d nth=%d (%d,%d) got=%d ref=%d\n",
                       M, K, N, nth, i, j, C[got_idx], ref[ref_idx]);
                return 0;
            }
        }
    }
    if (!check_i32_sentinel(C, pad_elems, guard))
        return 0;

    i8gemm_mt_dispatch_f(A_pad, B_reo, Cf, M, K_r, N_r, nth);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            size_t got_idx = (size_t)i * N_r + j;
            size_t ref_idx = (size_t)i * N + j;
            if (Cf[got_idx] != (float)ref[ref_idx]) {
                printf("I8 dispatch fp32 mismatch M=%d K=%d N=%d nth=%d (%d,%d)\n",
                       M, K, N, nth, i, j);
                return 0;
            }
        }
    }
    if (!check_f32_sentinel(Cf, pad_elems, guard))
        return 0;

    i8gemm_mt_dispatch_bias_f(A_pad, B_reo, Cbias, M, K_r, N_r, nth, bias_pad);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            size_t got_idx = (size_t)i * N_r + j;
            size_t ref_idx = (size_t)i * N + j;
            float expect = (float)ref[ref_idx] + bias[j];
            if (fabsf(Cbias[got_idx] - expect) > 1e-5f) {
                printf("I8 dispatch bias mismatch M=%d K=%d N=%d nth=%d (%d,%d)\n",
                       M, K, N, nth, i, j);
                return 0;
            }
        }
    }
    if (!check_f32_sentinel(Cbias, pad_elems, guard))
        return 0;

    free(A_pad);
    free(B_pad);
    free(B_reo);
    free(bias_pad);
    free(C);
    free(Cf);
    free(Cbias);
    return 1;
}

static int verify_bf16_case(int M, int K, int N, int nth) {
    const size_t elems = (size_t)M * N;
    const size_t guard = 32;
    bf16_t *A = (bf16_t *)malloc((size_t)M * K * sizeof(bf16_t));
    bf16_t *B = (bf16_t *)malloc((size_t)K * N * sizeof(bf16_t));
    float *bias = (float *)malloc((size_t)N * sizeof(float));
    float *ref = (float *)malloc(elems * sizeof(float));
    float *C = (float *)malloc((elems + guard) * sizeof(float));
    float *Cbias = (float *)malloc((elems + guard) * sizeof(float));
    bf16_t *Cbf16 = (bf16_t *)malloc((elems + guard) * sizeof(bf16_t));
    if (!A || !B || !bias || !ref || !C || !Cbias || !Cbf16)
        return 0;

    for (int i = 0; i < M * K; i++)
        A[i] = test_float_to_bf16(frand_range(-1.5f, 1.5f));
    for (int i = 0; i < K * N; i++)
        B[i] = test_float_to_bf16(frand_range(-1.5f, 1.5f));
    for (int i = 0; i < N; i++)
        bias[i] = frand_range(-2.0f, 2.0f);

    for (size_t i = 0; i < elems + guard; i++) {
        C[i] = SENTINEL_F32;
        Cbias[i] = SENTINEL_F32;
        Cbf16[i] = SENTINEL_BF16;
    }

    ref_bf16(A, B, NULL, ref, M, K, N);
    bf16gemm_mt(A, B, C, M, K, N, nth);
    for (size_t i = 0; i < elems; i++) {
        if (fabsf(C[i] - ref[i]) > BF16_EPSILON) {
            printf("BF16 fp32 mismatch M=%d K=%d N=%d nth=%d idx=%zu got=%f ref=%f\n",
                   M, K, N, nth, i, C[i], ref[i]);
            return 0;
        }
    }
    if (!check_f32_sentinel(C, elems, guard)) {
        printf("BF16 fp32 sentinel overwrite M=%d K=%d N=%d nth=%d\n", M, K, N, nth);
        return 0;
    }
    if (!verify_bf16_dispatch_case(A, B, bias, ref, M, K, N, nth))
        return 0;

    ref_bf16(A, B, bias, ref, M, K, N);
    bf16gemm_mt_bias_f(A, B, Cbias, M, K, N, nth, bias);
    for (size_t i = 0; i < elems; i++) {
        if (fabsf(Cbias[i] - ref[i]) > BF16_EPSILON) {
            printf("BF16 bias mismatch M=%d K=%d N=%d nth=%d idx=%zu got=%f ref=%f\n",
                   M, K, N, nth, i, Cbias[i], ref[i]);
            return 0;
        }
    }
    if (!check_f32_sentinel(Cbias, elems, guard)) {
        printf("BF16 bias sentinel overwrite M=%d K=%d N=%d nth=%d\n", M, K, N, nth);
        return 0;
    }

    ref_bf16(A, B, NULL, ref, M, K, N);
    bf16gemm_mt_nld_b(A, B, Cbf16, M, K, N, nth);
    for (size_t i = 0; i < elems; i++) {
        bf16_t expect = test_float_to_bf16(ref[i]);
        if (Cbf16[i] != expect) {
            float got_f = test_bf16_to_float(Cbf16[i]);
            float exp_f = test_bf16_to_float(expect);
            if (fabsf(got_f - exp_f) > BF16_EPSILON) {
                printf("BF16 nld_b mismatch M=%d K=%d N=%d nth=%d idx=%zu got=%f ref=%f\n",
                       M, K, N, nth, i, got_f, exp_f);
                return 0;
            }
        }
    }
    if (!check_bf16_sentinel(Cbf16, elems, guard)) {
        printf("BF16 nld_b sentinel overwrite M=%d K=%d N=%d nth=%d\n", M, K, N, nth);
        return 0;
    }

    free(A);
    free(B);
    free(bias);
    free(ref);
    free(C);
    free(Cbias);
    free(Cbf16);
    return 1;
}

static int verify_i8_case(int M, int K, int N, int nth) {
    const size_t elems = (size_t)M * N;
    const size_t guard = 32;
    int8_t *A = (int8_t *)malloc((size_t)M * K);
    int8_t *B = (int8_t *)malloc((size_t)K * N);
    float *bias = (float *)malloc((size_t)N * sizeof(float));
    int32_t *ref = (int32_t *)malloc(elems * sizeof(int32_t));
    int32_t *C = (int32_t *)malloc((elems + guard) * sizeof(int32_t));
    float *Cf = (float *)malloc((elems + guard) * sizeof(float));
    float *Cbias = (float *)malloc((elems + guard) * sizeof(float));
    if (!A || !B || !bias || !ref || !C || !Cf || !Cbias)
        return 0;

    for (int i = 0; i < M * K; i++)
        A[i] = (int8_t)((rand() % 17) - 8);
    for (int i = 0; i < K * N; i++)
        B[i] = (int8_t)((rand() % 17) - 8);
    for (int i = 0; i < N; i++)
        bias[i] = frand_range(-4.0f, 4.0f);

    for (size_t i = 0; i < elems + guard; i++) {
        C[i] = SENTINEL_I32;
        Cf[i] = SENTINEL_F32;
        Cbias[i] = SENTINEL_F32;
    }

    ref_i8(A, B, ref, M, K, N);
    i8gemm_mt(A, B, C, M, K, N, nth);
    for (size_t i = 0; i < elems; i++) {
        if (C[i] != ref[i]) {
            printf("I8 int32 mismatch M=%d K=%d N=%d nth=%d idx=%zu got=%d ref=%d\n",
                   M, K, N, nth, i, C[i], ref[i]);
            return 0;
        }
    }
    if (!check_i32_sentinel(C, elems, guard)) {
        printf("I8 int32 sentinel overwrite M=%d K=%d N=%d nth=%d\n", M, K, N, nth);
        return 0;
    }
    if (!verify_i8_dispatch_case(A, B, bias, ref, M, K, N, nth))
        return 0;

    i8gemm_mt_f(A, B, Cf, M, K, N, nth);
    for (size_t i = 0; i < elems; i++) {
        if (Cf[i] != (float)ref[i]) {
            printf("I8 fp32 mismatch M=%d K=%d N=%d nth=%d idx=%zu got=%f ref=%d\n",
                   M, K, N, nth, i, Cf[i], ref[i]);
            return 0;
        }
    }
    if (!check_f32_sentinel(Cf, elems, guard)) {
        printf("I8 fp32 sentinel overwrite M=%d K=%d N=%d nth=%d\n", M, K, N, nth);
        return 0;
    }

    i8gemm_mt_bias_f(A, B, Cbias, M, K, N, nth, bias);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            size_t idx = (size_t)i * N + j;
            float expect = (float)ref[idx] + bias[j];
            if (fabsf(Cbias[idx] - expect) > 1e-5f) {
                printf("I8 bias mismatch M=%d K=%d N=%d nth=%d idx=%zu got=%f ref=%f\n",
                       M, K, N, nth, idx, Cbias[idx], expect);
                return 0;
            }
        }
    }
    if (!check_f32_sentinel(Cbias, elems, guard)) {
        printf("I8 bias sentinel overwrite M=%d K=%d N=%d nth=%d\n", M, K, N, nth);
        return 0;
    }

    free(A);
    free(B);
    free(bias);
    free(ref);
    free(C);
    free(Cf);
    free(Cbias);
    return 1;
}

int main(void) {
    const int Ms[] = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16};
    const int Ks[] = {1, 2, 7, 8, 9, 15, 16, 17, 31, 32};
    const int Ns[] = {1, 2, 7, 8, 9, 15, 16, 17, 31, 32};
    const int threads[] = {1, 2, 4};
    size_t cases = 0;

    srand(20260605);
    for (size_t ti = 0; ti < sizeof(threads) / sizeof(threads[0]); ti++) {
        for (size_t mi = 0; mi < sizeof(Ms) / sizeof(Ms[0]); mi++) {
            for (size_t ki = 0; ki < sizeof(Ks) / sizeof(Ks[0]); ki++) {
                for (size_t ni = 0; ni < sizeof(Ns) / sizeof(Ns[0]); ni++) {
                    int M = Ms[mi];
                    int K = Ks[ki];
                    int N = Ns[ni];
                    int nth = threads[ti];
                    if (!verify_bf16_case(M, K, N, nth))
                        return 1;
                    if (!verify_i8_case(M, K, N, nth))
                        return 1;
                    cases += 2;
                }
            }
        }
    }

    printf("SVE correctness OK: %zu wrapper+dispatch cases\n", cases);
    return 0;
}
