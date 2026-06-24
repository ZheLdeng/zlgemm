// Measure the experimental optimized i8 8x16 microkernel, pre-packed, in L1.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int8_t i8_t;
typedef int32_t i32_t;

void i8_pack_B(const i8_t *B, i8_t *B_reo, int K, int N);
void i8_pack_A_neon_m8_asm(const i8_t *A, i8_t *P, int K_r, int lda);
void i8_kernel_opt(const i8_t *A_pack, const i8_t *B_pack, i32_t *C, int K);

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static void *aa(size_t n){ void*p=aligned_alloc(64,(n+63)&~63ull); memset(p,1,(n+63)&~63ull); return p; }

int main(void) {
    const double i8_peak = 661.22;
    int M = 8, K = 1024, N = 16;
    i8_t *A = aa((size_t)M*K), *B = aa((size_t)K*N);
    // pad packed buffers by one extra K16 block to absorb the prologue over-read
    i8_t *B_reo = aa((size_t)K*N + 4096);
    i8_t *A_reo = aa((size_t)K*8 + 4096);
    i32_t *C = aa((size_t)M*N*sizeof(i32_t)*64);
    for (int i=0;i<M*K;i++) A[i]=(i%17)-8;
    for (int i=0;i<K*N;i++) B[i]=(i%13)-6;
    i8_pack_B(B,B_reo,K,N);
    i8_pack_A_neon_m8_asm(A,A_reo,K,K);

    long reps = 400000;
    for (int w=0;w<1000;w++) i8_kernel_opt(A_reo,B_reo,C,K);
    double best=1e300;
    for (int r=0;r<5;r++){ double t0=now_sec(); for(long i=0;i<reps;i++) i8_kernel_opt(A_reo,B_reo,C,K); double dt=now_sec()-t0; if(dt<best)best=dt; }
    double ops = 2.0*M*K*N*(double)reps;
    double g = ops/best/1e9;
    printf("opt i8 8x16 unrolled (M8 N16 K%d, in-L1): %8.2f GOPS = %.1f%% of peak\n", K, g, g*100.0/i8_peak);
    return 0;
}
