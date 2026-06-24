#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gemm_params.h"
typedef int8_t i8_t; typedef int32_t i32_t;
void i8_pack_B(const i8_t*,i8_t*,int,int);
void i8gemm_k_ld(const i8_t*,const i8_t*,i32_t*,i8_t*,const gemm_params_t*);
static double ns(){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void*aa(size_t n){void*p=aligned_alloc(64,(n+63)&~63ull);memset(p,1,(n+63)&~63ull);return p;}
static int rup(int x,int q){return ((x+q-1)/q)*q;}
int main(int argc,char**argv){
    int M=8, K=argc>1?atoi(argv[1]):1024, N=argc>2?atoi(argv[2]):64;
    int Kr=rup(K,8), Nr=rup(N,8);
    i8_t*A=aa((size_t)M*Kr),*B=aa((size_t)Kr*Nr),*Br=aa((size_t)Kr*Nr),*Areo=aa((size_t)M*Kr+4096);
    i32_t*C=aa((size_t)M*Nr*sizeof(i32_t)*4);
    for(int i=0;i<M*Kr;i++)A[i]=(i%17)-8;
    for(int i=0;i<Kr*Nr;i++)B[i]=(i%13)-6;
    i8_pack_B(B,Br,Kr,Nr);
    gemm_params_t p={M,Kr,Nr,Kr,Kr,Nr};
    long reps=400000;
    for(int w=0;w<1000;w++) i8gemm_k_ld(A,Br,C,Areo,&p);
    double best=1e30;
    for(int r=0;r<5;r++){double t=ns();for(long j=0;j<reps;j++)i8gemm_k_ld(A,Br,C,Areo,&p);double d=ns()-t;if(d<best)best=d;}
    double g=2.0*M*K*N*(double)reps/best/1e9;
    printf("NEON i8 k_ld (M8 N%d K%d in-L1): %.1f GOPS = %.1f%% of peak(661)\n",N,K,g,g*100/661.22);
    return 0;
}
