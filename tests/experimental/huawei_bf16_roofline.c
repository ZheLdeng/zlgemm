// huawei_bf16_roofline.c -- drive the bfmmla accumulator-count sweep.
//
// Reports per-core GFLOPS (2-op) and bfmmla/cycle. Answers: is the SVE bf16
// single-core ceiling (~48% of the assumed 185.4 peak in the fine sweep) due to
// the bfmmla issue rate itself (this roofline also caps ~48% -> peak is really
// ~93, kernel is fine) or to kernel load/loop overhead (this roofline hits
// ~1 bfmmla/cycle = ~185 GFLOPS -> kernel has headroom).
//
// Usage: ./huawei_bf16_roofline [GHZ]   (Huawei 2.9, V1 2.6)
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void roof_bf16_8(uint64_t);
void roof_bf16_16(uint64_t);
void roof_bf16_24(uint64_t);
void roof_bf16_30(uint64_t);

static double now_sec(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);
    return (double)t.tv_sec+(double)t.tv_nsec*1e-9;}

static double best_of(void(*fn)(uint64_t),uint64_t it){
    fn(it/10);
    double b=1e300;
    for(int r=0;r<6;r++){double t=now_sec();fn(it);double d=now_sec()-t;if(d<b)b=d;}
    return b;
}

int main(int argc,char**argv){
    double ghz = argc>1 ? atof(argv[1]) : 2.9;
    const uint64_t it = 40000000ull;
    struct { const char*name; void(*fn)(uint64_t); int n; int macs; } v[] = {
        {"bf16_acc8 ", roof_bf16_8 ,  8, 32},
        {"bf16_acc16", roof_bf16_16, 16, 32},
        {"bf16_acc24", roof_bf16_24, 24, 32},
        {"bf16_acc30", roof_bf16_30, 30, 32},
    };
    printf("# bfmmla roofline  (GHz=%.2f, pin with taskset -c <core>)\n", ghz);
    printf("# peak @1 bfmmla/cycle = %.1f GFLOPS(2op)\n", 32.0*2.0*ghz);
    printf("%-10s %12s %12s %14s\n","kernel","GFLOPS(2op)","GMAC/s","bfmmla/cycle");
    for(unsigned i=0;i<sizeof v/sizeof v[0];i++){
        double s=best_of(v[i].fn,it);
        double inst = (double)v[i].n*(double)it;
        double gmac  = inst*v[i].macs/s/1e9;
        double gflop2= gmac*2.0;
        double ipc   = inst/s/(ghz*1e9);
        printf("%-10s %12.1f %12.1f %14.3f\n", v[i].name, gflop2, gmac, ipc);
    }
    return 0;
}
