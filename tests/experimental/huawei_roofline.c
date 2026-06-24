// huawei_roofline.c -- drive the smmla accumulator-count sweep.
//
// Reports per-core GOPS (2-op) and smmla/cycle (given --ghz). On Huawei this
// answers the open question from huawei_80c_scheduling_progress: is the SVE
// smmla compute ceiling latency-bound (rises with #acc) or throughput-capped
// (flat), and how it compares to the NEON smmla peak that cpufb measured.
//
// Usage: ./huawei_roofline [GHZ]   (default GHz=2.9 for Huawei; V1 use 2.6)
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void roof_sve_8(uint64_t);
void roof_sve_16(uint64_t);
void roof_sve_24(uint64_t);
void roof_sve_30(uint64_t);
void roof_neon_24(uint64_t);

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
        {"sve_acc8 ", roof_sve_8 ,  8, 64},
        {"sve_acc16", roof_sve_16, 16, 64},
        {"sve_acc24", roof_sve_24, 24, 64},
        {"sve_acc30", roof_sve_30, 30, 64},
        {"neon_acc24",roof_neon_24,24, 32},
    };
    printf("# smmla roofline  (GHz=%.2f, pin with taskset -c <core>)\n", ghz);
    printf("%-10s %12s %12s %14s\n","kernel","GOPS(2op)","GMAC/s","smmla/cycle");
    for(unsigned i=0;i<sizeof v/sizeof v[0];i++){
        double s=best_of(v[i].fn,it);
        double smmla = (double)v[i].n*(double)it;     // total smmla issued
        double gmac  = smmla*v[i].macs/s/1e9;
        double gops2 = gmac*2.0;
        double spc   = smmla/s/(ghz*1e9);
        printf("%-10s %12.1f %12.1f %14.3f\n", v[i].name, gops2, gmac, spc);
    }
    return 0;
}
