// bench_nld_sve.c -- compute-only BF16 SVE kernel probe.

#include <arm_sve.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

void bf16gemm_sve_nld_compute(uint64_t k4_iters);

static double elapsed_sec(const struct timespec *s, const struct timespec *e) {
    return (e->tv_sec - s->tv_sec) + (e->tv_nsec - s->tv_nsec) * 1e-9;
}

static int sve_segments(void) {
    return (int)(svcntb() / 16);
}

static double measure_peak_sve_bfmmla(void) {
    volatile int64_t n;
    const int64_t iters = 10000000LL;
    const double ops_per_iter = 16.0 * 32.0 * (double)sve_segments();
    double best = 0.0;
    struct timespec s, e;

#define SVE_BFM_BODY \
    "bfmmla z0.s, z16.h, z24.h\n\t" \
    "bfmmla z1.s, z17.h, z25.h\n\t" \
    "bfmmla z2.s, z18.h, z26.h\n\t" \
    "bfmmla z3.s, z19.h, z27.h\n\t" \
    "bfmmla z4.s, z20.h, z28.h\n\t" \
    "bfmmla z5.s, z21.h, z29.h\n\t" \
    "bfmmla z6.s, z22.h, z30.h\n\t" \
    "bfmmla z7.s, z23.h, z31.h\n\t" \
    "bfmmla z8.s, z16.h, z24.h\n\t" \
    "bfmmla z9.s, z17.h, z25.h\n\t" \
    "bfmmla z10.s, z18.h, z26.h\n\t" \
    "bfmmla z11.s, z19.h, z27.h\n\t" \
    "bfmmla z12.s, z20.h, z28.h\n\t" \
    "bfmmla z13.s, z21.h, z29.h\n\t" \
    "bfmmla z14.s, z22.h, z30.h\n\t" \
    "bfmmla z15.s, z23.h, z31.h\n\t"

    for (int run = 0; run < 5; run++) {
        n = iters;
        clock_gettime(CLOCK_MONOTONIC_RAW, &s);
        __asm__ volatile(
            "dup z16.h, #1\n\t"
            "dup z17.h, #1\n\t"
            "dup z18.h, #1\n\t"
            "dup z19.h, #1\n\t"
            "dup z20.h, #1\n\t"
            "dup z21.h, #1\n\t"
            "dup z22.h, #1\n\t"
            "dup z23.h, #1\n\t"
            "dup z24.h, #1\n\t"
            "dup z25.h, #1\n\t"
            "dup z26.h, #1\n\t"
            "dup z27.h, #1\n\t"
            "dup z28.h, #1\n\t"
            "dup z29.h, #1\n\t"
            "dup z30.h, #1\n\t"
            "dup z31.h, #1\n\t"
            "mov z0.s, #0\n\t"
            "mov z1.s, #0\n\t"
            "mov z2.s, #0\n\t"
            "mov z3.s, #0\n\t"
            "mov z4.s, #0\n\t"
            "mov z5.s, #0\n\t"
            "mov z6.s, #0\n\t"
            "mov z7.s, #0\n\t"
            "mov z8.s, #0\n\t"
            "mov z9.s, #0\n\t"
            "mov z10.s, #0\n\t"
            "mov z11.s, #0\n\t"
            "mov z12.s, #0\n\t"
            "mov z13.s, #0\n\t"
            "mov z14.s, #0\n\t"
            "mov z15.s, #0\n\t"
            "1:\n\t"
            SVE_BFM_BODY
            "subs %[n], %[n], #1\n\t"
            "b.ne 1b\n\t"
            : [n] "+r"(n)
            :
            : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
              "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
              "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
              "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
              "cc", "memory");
        clock_gettime(CLOCK_MONOTONIC_RAW, &e);
        double sec = elapsed_sec(&s, &e);
        if (sec > 0.0) {
            double gflops = (double)iters * ops_per_iter / sec * 1e-9;
            if (gflops > best)
                best = gflops;
        }
    }
#undef SVE_BFM_BODY
    return best;
}

static double bench_nld_compute(uint64_t k4_iters) {
    const double ops_per_iter = 24.0 * 32.0 * (double)sve_segments();
    double best = 0.0;
    struct timespec s, e;

    for (int run = 0; run < 10; run++) {
        clock_gettime(CLOCK_MONOTONIC_RAW, &s);
        bf16gemm_sve_nld_compute(k4_iters);
        clock_gettime(CLOCK_MONOTONIC_RAW, &e);
        double sec = elapsed_sec(&s, &e);
        if (sec > 0.0) {
            double gflops = (double)k4_iters * ops_per_iter / sec * 1e-9;
            if (gflops > best)
                best = gflops;
        }
    }
    return best;
}

int main(void) {
    const uint64_t k4_iters = 20000000ULL;
    double peak = measure_peak_sve_bfmmla();
    double nld = bench_nld_compute(k4_iters);
    double eff = peak > 0.0 ? nld / peak * 100.0 : 0.0;

    printf("SVE VL=%zu bits, 128b segments=%d\n", svcntb() * 8, sve_segments());
    printf("bfmmla_peak_GFLOPS,compute_only_GFLOPS,compute_only_eff_pct\n");
    printf("%.2f,%.2f,%.2f\n", peak, nld, eff);
    return 0;
}
