// compute_ceiling.c -- measure pure-compute ceiling vs cpufb peak.
#include <stdint.h>
#include <stdio.h>
#include <time.h>

void compute_only_i8_16(uint64_t iters);
void compute_only_bf16_16(uint64_t iters);
void compute_ld_i8_16(uint64_t iters, const void *buf);
void compute_ld_i8_24(uint64_t iters, const void *buf);

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(void) {
    const uint64_t iters = 200000000ull;   // 200M loop iterations
    // Per iteration: 16 mmla. Each mmla on 256-bit SVE = 2 segments.
    //   i8 smmla:  2 * (2*2*8) = 64 MACs  -> 128 ops (2 ops/MAC)
    //   bf16 bfmmla: 2 * (2*2*4) = 32 MACs -> 64 ops
    const double i8_ops_per_iter = 16.0 * 128.0;
    const double bf16_ops_per_iter = 16.0 * 64.0;

    // peak per core, harness units (2 ops/MAC):
    const double i8_peak = 661.22;   // 330.61 GOPS(MAC) * 2
    const double bf16_peak = 330.96; // 165.48 GFLOPS(MAC) * 2

    // warmup
    compute_only_i8_16(iters / 10);
    compute_only_bf16_16(iters / 10);

    double best = 1e300;
    for (int r = 0; r < 5; r++) {
        double t0 = now_sec();
        compute_only_i8_16(iters);
        double dt = now_sec() - t0;
        if (dt < best) best = dt;
    }
    double i8_gops = i8_ops_per_iter * (double)iters / best / 1e9;

    best = 1e300;
    for (int r = 0; r < 5; r++) {
        double t0 = now_sec();
        compute_only_bf16_16(iters);
        double dt = now_sec() - t0;
        if (dt < best) best = dt;
    }
    double bf16_gflops = bf16_ops_per_iter * (double)iters / best / 1e9;

    printf("compute-only i8  smmla : %8.2f GOPS   = %.1f%% of peak (%.2f)\n",
           i8_gops, i8_gops * 100.0 / i8_peak, i8_peak);
    printf("compute-only bf16 bfmmla: %8.2f GFLOPS = %.1f%% of peak (%.2f)\n",
           bf16_gflops, bf16_gflops * 100.0 / bf16_peak, bf16_peak);

    // compute + load variants (L1-resident scratch)
    void *buf = aligned_alloc(64, 4096);
    memset(buf, 1, 4096);
    best = 1e300;
    for (int r = 0; r < 5; r++) {
        double t0 = now_sec(); compute_ld_i8_16(iters, buf); double dt = now_sec() - t0;
        if (dt < best) best = dt;
    }
    double ld16 = i8_ops_per_iter * (double)iters / best / 1e9;
    best = 1e300;
    for (int r = 0; r < 5; r++) {
        double t0 = now_sec(); compute_ld_i8_24(iters, buf); double dt = now_sec() - t0;
        if (dt < best) best = dt;
    }
    double ld24 = 24.0 * 128.0 * (double)iters / best / 1e9;
    printf("cmp+load i8 4x4 (16 mmla+8 ld, intens 2.0): %8.2f GOPS = %.1f%% of peak\n",
           ld16, ld16 * 100.0 / i8_peak);
    printf("cmp+load i8 8x3 (24 mmla+10 ld, intens 2.4): %8.2f GOPS = %.1f%% of peak\n",
           ld24, ld24 * 100.0 / i8_peak);
    return 0;
}
