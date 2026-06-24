#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gemm_params.h"
typedef int8_t i8_t; typedef int32_t i32_t;
void i8_pack_B(const i8_t*,i8_t*,int,int);
void i8gemm_k_hybrid(const i8_t*,const i8_t*,i32_t*,i8_t*,const gemm_params_t*);
void i8gemm_mt_dispatch(const i8_t*,const i8_t*,i32_t*,int,int,int,int);
static double ns(){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int rup(int x,int q){return ((x+q-1)/q)*q;}
static void*aa(size_t n){void*p=aligned_alloc(64,(n+63)&~63ull);memset(p,1,(n+63)&~63ull);return p;}

int main(int argc,char**argv){
    struct{int M,K,N;}cs[]={{16,16,16},{32,32,32},{48,48,48},{64,64,64},{96,96,96},
                            {128,128,128},{192,192,192},{256,256,256},{8,256,256},{8,512,512}};
    double peak=661.22;
    printf("%-14s %10s %10s %10s\n","shape","hybrid","dispatch","hyb%pk");
    for(unsigned i=0;i<sizeof cs/sizeof cs[0];i++){
        int M=cs[i].M,K=cs[i].K,N=cs[i].N, Kr=rup(K,16),Nr=rup(N,16);
        i8_t*A=aa((size_t)M*Kr),*B=aa((size_t)Kr*Nr),*Br=aa((size_t)Kr*Nr);
        i32_t*C=aa((size_t)M*Nr*sizeof(i32_t));
        i8_pack_B(B,Br,Kr,Nr);
        gemm_params_t p={M,Kr,Nr,Kr,Kr,Nr};
        long reps=2000000/(M*K*N/4096+1);
        for(int w=0;w<2000;w++) i8gemm_k_hybrid(A,Br,C,NULL,&p);
        double bh=1e30; for(int r=0;r<5;r++){double t=ns();for(long j=0;j<reps;j++)i8gemm_k_hybrid(A,Br,C,NULL,&p);double d=ns()-t;if(d<bh)bh=d;}
        for(int w=0;w<2000;w++) i8gemm_mt_dispatch(A,Br,C,M,Kr,Nr,1);
        double bd=1e30; for(int r=0;r<5;r++){double t=ns();for(long j=0;j<reps;j++)i8gemm_mt_dispatch(A,Br,C,M,Kr,Nr,1);double d=ns()-t;if(d<bd)bd=d;}
        double ops=2.0*M*K*N*reps;
        double gh=ops/bh/1e9, gd=ops/bd/1e9;
        char s[32]; snprintf(s,32,"%dx%dx%d",M,K,N);
        printf("%-14s %10.1f %10.1f %9.0f%%\n",s,gh,gd,gh*100/peak);
        free(A);free(B);free(Br);free(C);
    }
    return 0;
}
