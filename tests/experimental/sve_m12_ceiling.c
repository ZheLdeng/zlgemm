#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gemm_params.h"
typedef int8_t i8_t; typedef int32_t i32_t;
void i8_pack_B(const i8_t*,i8_t*,int,int);
void i8gemm_k_nld_m12(const i8_t*,const i8_t*,i32_t*,i8_t*,const gemm_params_t*);
static double ns(){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void*aa(size_t n){void*p=aligned_alloc(64,(n+63)&~63ull);memset(p,1,(n+63)&~63ull);return p;}
static void packA(const i8_t*A,i8_t*R,int Kr){size_t idx=0;for(int kb=0;kb<Kr;kb+=8)for(int rp=0;rp<6;rp++){int r0=rp*2,r1=r0+1;memcpy(R+idx,A+(size_t)r0*Kr+kb,8);idx+=8;memcpy(R+idx,A+(size_t)r1*Kr+kb,8);idx+=8;}}
int main(int argc,char**argv){
    int M=12,K=argc>1?atoi(argv[1]):1024,N=16;
    i8_t*B=aa((size_t)K*N),*Br=aa((size_t)K*N),*Af=aa((size_t)M*K),*Ar=aa((size_t)K*12+64);
    i32_t*C=aa((size_t)M*N*sizeof(i32_t)*64);
    for(int i=0;i<M*K;i++)Af[i]=(i%17)-8;
    for(int i=0;i<K*N;i++)B[i]=(i%13)-6;
    i8_pack_B(B,Br,K,N); packA(Af,Ar,K);
    gemm_params_t p={M,K,N,K,K,N};
    long reps=400000;
    for(int w=0;w<1000;w++)i8gemm_k_nld_m12(Af,Br,C,Ar,&p);
    double best=1e30;for(int r=0;r<5;r++){double t=ns();for(long j=0;j<reps;j++)i8gemm_k_nld_m12(Af,Br,C,Ar,&p);double d=ns()-t;if(d<best)best=d;}
    double g=2.0*M*K*N*(double)reps/best/1e9;
    printf("SVE m12 prepacked (M12 N16 K%d in-L1): %.1f GOPS = %.1f%% of peak(661)\n",K,g,g*100/661.22);
    return 0;
}
