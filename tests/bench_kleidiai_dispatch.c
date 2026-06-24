// bench_kleidiai_dispatch.c -- fixed-kernel KleidiAI benchmark adapter.
//
// BF16: kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla
// I8:   kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm
//
// The adapter deliberately bypasses KleidiAI registries and calls the exact
// micro-kernel symbols named above. B/RHS is packed once outside timing. C/D is
// row-major. Parallelism is an outer OpenMP schedule over M blocks.

#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <omp.h>

#include "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_bf16p8x4_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi8cxp_qsi8cx_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int env_flag(const char *name, int fallback) {
    const char *s = getenv(name);
    if (!s || !*s) return fallback;
    return atoi(s) != 0;
}

static size_t round64(size_t n) {
    return (n + 63u) & ~(size_t)63u;
}

static void *xalloc(size_t n) {
    if (n == 0) n = 64;
    void *p = aligned_alloc(64, round64(n));
    if (!p) {
        perror("aligned_alloc");
        exit(1);
    }
    memset(p, 0, round64(n));
    return p;
}

static float value_f32(size_t i, int salt) {
    return (float)(((int)((i + (size_t)salt) % 17) - 8)) * 0.125f;
}

static const char *cache_class(double kib) {
    if (kib <= 96.0) return "L1";
    if (kib <= 640.0) return "H2";
    if (kib <= 1280.0) return "L2";
    return "GT_L2";
}

static void print_row(const char *impl, const char *dtype, int M, int K, int N, int threads, double kib, int reps,
                      double perf, const char *status, const char *note) {
    printf("%s,%s,%s,%d,%d,%d,%d,%.1f,%d,%.3f,%s,%s\n", impl, dtype, cache_class(kib), M, K, N, threads, kib, reps,
           perf, status, note);
}

static double bench_bf16(int M, int K, int N, int reps, int warmup, int runs, int threads, double *kib_out) {
    const size_t mr = kai_get_mr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla();
    const size_t nr = kai_get_nr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla();
    const size_t kr = kai_get_kr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla();
    const size_t sr = kai_get_sr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla();
    const size_t m_step = kai_get_m_step_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla();

    float *lhs = (float *)xalloc((size_t)M * (size_t)K * sizeof(float));
    float *rhs = (float *)xalloc((size_t)K * (size_t)N * sizeof(float));
    float *bias = (float *)xalloc((size_t)N * sizeof(float));
    float *dst = (float *)xalloc((size_t)M * (size_t)N * sizeof(float));
    for (size_t i = 0; i < (size_t)M * (size_t)K; ++i) lhs[i] = value_f32(i, 1);
    for (size_t i = 0; i < (size_t)K * (size_t)N; ++i) rhs[i] = value_f32(i, 2);

    const size_t lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_bf16p8x4_f32_neon(M, K, mr, kr, sr);
    const size_t rhs_packed_size = kai_get_rhs_packed_size_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon(N, K, nr, kr);
    uint8_t *lhs_packed = (uint8_t *)xalloc(lhs_packed_size);
    uint8_t *rhs_packed = (uint8_t *)xalloc(rhs_packed_size);

    kai_run_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon(1, N, K, nr, kr, sr, (size_t)N * sizeof(float), rhs, bias,
                                                         NULL, rhs_packed, 0, NULL);

    *kib_out = ((double)lhs_packed_size + (double)rhs_packed_size + (double)M * N * sizeof(float)) / 1024.0;
    omp_set_num_threads(threads);
    const int lhs_pack_in_timing = env_flag("KLEIDIAI_LHS_PACK_IN_TIMING", 1);

    if (!lhs_pack_in_timing)
        kai_run_lhs_quant_pack_bf16p8x4_f32_neon(M, K, mr, kr, sr, 0, lhs, (size_t)K * sizeof(float), lhs_packed);

    for (int w = 0; w < warmup; ++w) {
        if (lhs_pack_in_timing)
            kai_run_lhs_quant_pack_bf16p8x4_f32_neon(M, K, mr, kr, sr, 0, lhs, (size_t)K * sizeof(float), lhs_packed);
#pragma omp parallel for schedule(static)
        for (size_t m_idx = 0; m_idx < (size_t)M; m_idx += m_step) {
            const size_t height = ((size_t)M - m_idx < m_step) ? ((size_t)M - m_idx) : m_step;
            kai_run_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla(
                height, N, K, lhs_packed + kai_get_lhs_packed_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla(m_idx, K),
                rhs_packed, dst + m_idx * (size_t)N, (size_t)N * sizeof(float), sizeof(float), -FLT_MAX, FLT_MAX);
        }
    }

    double best = 0.0;
    const double ops = 2.0 * (double)M * (double)K * (double)N;
    for (int r = 0; r < runs; ++r) {
        const double t0 = now_sec();
        for (int rep = 0; rep < reps; ++rep) {
            if (lhs_pack_in_timing)
                kai_run_lhs_quant_pack_bf16p8x4_f32_neon(M, K, mr, kr, sr, 0, lhs, (size_t)K * sizeof(float), lhs_packed);
#pragma omp parallel for schedule(static)
            for (size_t m_idx = 0; m_idx < (size_t)M; m_idx += m_step) {
                const size_t height = ((size_t)M - m_idx < m_step) ? ((size_t)M - m_idx) : m_step;
                kai_run_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla(
                    height, N, K, lhs_packed + kai_get_lhs_packed_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla(m_idx, K),
                    rhs_packed, dst + m_idx * (size_t)N, (size_t)N * sizeof(float), sizeof(float), -FLT_MAX, FLT_MAX);
            }
        }
        const double dt = (now_sec() - t0) / (double)reps;
        const double perf = ops / dt / 1e9;
        if (perf > best) best = perf;
    }

    free(lhs);
    free(rhs);
    free(bias);
    free(dst);
    free(lhs_packed);
    free(rhs_packed);
    return best;
}

static double bench_i8(int M, int K, int N, int reps, int warmup, int runs, int threads, double *kib_out) {
    const size_t mr = kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm();
    const size_t nr = kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm();
    const size_t kr = kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm();
    const size_t sr = kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm();
    const size_t m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm();

    float *lhs = (float *)xalloc((size_t)M * (size_t)K * sizeof(float));
    int8_t *rhs = (int8_t *)xalloc((size_t)K * (size_t)N);
    float *bias = (float *)xalloc((size_t)N * sizeof(float));
    float *scale = (float *)xalloc((size_t)N * sizeof(float));
    float *dst = (float *)xalloc((size_t)M * (size_t)N * sizeof(float));
    for (size_t i = 0; i < (size_t)M * (size_t)K; ++i) lhs[i] = value_f32(i, 1);
    for (size_t i = 0; i < (size_t)K * (size_t)N; ++i) rhs[i] = (int8_t)(((int)((i + 2) % 17) - 8));
    for (int i = 0; i < N; ++i) scale[i] = 1.0f;

    const size_t lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr);
    const size_t rhs_packed_size = kai_get_rhs_packed_size_rhs_pack_kxn_qsi8cxp_qsi8cx_neon(N, K, nr, kr, sr);
    uint8_t *lhs_packed = (uint8_t *)xalloc(lhs_packed_size);
    uint8_t *rhs_packed = (uint8_t *)xalloc(rhs_packed_size);

    const struct kai_rhs_pack_qsi8cx_params params = {.lhs_zero_point = 1, .scale_multiplier = 1.0f};
    kai_run_rhs_pack_kxn_qsi8cxp_qsi8cx_neon(1, N, K, nr, kr, sr, rhs, bias, scale, rhs_packed, 0, &params);

    *kib_out = ((double)lhs_packed_size + (double)rhs_packed_size + (double)M * N * sizeof(float)) / 1024.0;
    omp_set_num_threads(threads);
    const int lhs_pack_in_timing = env_flag("KLEIDIAI_LHS_PACK_IN_TIMING", 1);

    if (!lhs_pack_in_timing)
        kai_run_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr, 0, lhs, (size_t)K * sizeof(float), lhs_packed);

    for (int w = 0; w < warmup; ++w) {
        if (lhs_pack_in_timing)
            kai_run_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr, 0, lhs, (size_t)K * sizeof(float), lhs_packed);
#pragma omp parallel for schedule(static)
        for (size_t m_idx = 0; m_idx < (size_t)M; m_idx += m_step) {
            const size_t height = ((size_t)M - m_idx < m_step) ? ((size_t)M - m_idx) : m_step;
            kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(
                height, N, K, lhs_packed + kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(m_idx, K),
                rhs_packed, dst + m_idx * (size_t)N, (size_t)N * sizeof(float), sizeof(float), -FLT_MAX, FLT_MAX);
        }
    }

    double best = 0.0;
    const double ops = 2.0 * (double)M * (double)K * (double)N;
    for (int r = 0; r < runs; ++r) {
        const double t0 = now_sec();
        for (int rep = 0; rep < reps; ++rep) {
            if (lhs_pack_in_timing)
                kai_run_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr, 0, lhs, (size_t)K * sizeof(float), lhs_packed);
#pragma omp parallel for schedule(static)
            for (size_t m_idx = 0; m_idx < (size_t)M; m_idx += m_step) {
                const size_t height = ((size_t)M - m_idx < m_step) ? ((size_t)M - m_idx) : m_step;
                kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(
                    height, N, K, lhs_packed + kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(m_idx, K),
                    rhs_packed, dst + m_idx * (size_t)N, (size_t)N * sizeof(float), sizeof(float), -FLT_MAX, FLT_MAX);
            }
        }
        const double dt = (now_sec() - t0) / (double)reps;
        const double perf = ops / dt / 1e9;
        if (perf > best) best = perf;
    }

    free(lhs);
    free(rhs);
    free(bias);
    free(scale);
    free(dst);
    free(lhs_packed);
    free(rhs_packed);
    return best;
}

int main(int argc, char **argv) {
    if (argc != 9) {
        fprintf(stderr, "usage: %s bf16|i8 M K N reps warmup runs threads\n", argv[0]);
        return 2;
    }
    const char *dtype = argv[1];
    const int M = atoi(argv[2]);
    const int K = atoi(argv[3]);
    const int N = atoi(argv[4]);
    const int reps = atoi(argv[5]);
    const int warmup = atoi(argv[6]);
    const int runs = atoi(argv[7]);
    const int threads = atoi(argv[8]);

    double kib = 0.0;
    double perf = 0.0;
    const char *impl = "";
    const char *note = "";
    if (strcmp(dtype, "bf16") == 0) {
        impl = "kleidiai_bf16_neon_mmla";
        note = "fixed=kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla; instr=bfmmla; omp=m_blocks";
        perf = bench_bf16(M, K, N, reps, warmup, runs, threads, &kib);
    } else if (strcmp(dtype, "i8") == 0) {
        impl = "kleidiai_i8_neon_i8mm";
        note = "fixed=kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm; instr=smmla; output=f32_dequant; omp=m_blocks";
        perf = bench_i8(M, K, N, reps, warmup, runs, threads, &kib);
    } else {
        print_row("kleidiai_fixed", dtype, M, K, N, threads, 0.0, reps, 0.0, "unsupported", "bad dtype");
        return 0;
    }
    print_row(impl, dtype, M, K, N, threads, kib, reps, perf, "ok", note);
    return 0;
}
