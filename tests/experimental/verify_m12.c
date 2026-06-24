// Verify i8 GEMM (incl. m12 path) against a naive int32 reference.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int8_t i8_t; typedef int32_t i32_t;
void i8gemm_mt(const i8_t*, const i8_t*, i32_t*, int M,int K,int N,int threads);

static int check(int M,int K,int N,int th){
    i8_t *A=malloc((size_t)M*K), *B=malloc((size_t)K*N);
    i32_t *C=calloc((size_t)M*N,sizeof(i32_t)), *R=calloc((size_t)M*N,sizeof(i32_t));
    for(int i=0;i<M*K;i++)A[i]=(int8_t)((i*7+3)%17-8);
    for(int i=0;i<K*N;i++)B[i]=(int8_t)((i*5+1)%13-6);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){long s=0;for(int k=0;k<K;k++)s+=(int)A[(size_t)m*K+k]*(int)B[(size_t)k*N+n];R[(size_t)m*N+n]=(i32_t)s;}
    i8gemm_mt(A,B,C,M,K,N,th);
    int bad=0; for(size_t i=0;i<(size_t)M*N && bad<5;i++) if(C[i]!=R[i]){printf("  MISMATCH @%zu got %d exp %d\n",i,C[i],R[i]);bad++;}
    free(A);free(B);free(C);free(R);
    return bad==0;
}
int main(int argc,char**argv){
    int th=argc>1?atoi(argv[1]):1;
    struct{int M,K,N;}cases[]={{96,96,96},{12,96,96},{24,128,64},{256,256,256},{120,512,128},
                               {12,64,32},{36,256,48},{100,96,96},{13,80,17},{96,96,16},{8,512,512},{8,256,64},{5,128,48},{7,1024,96},{4,512,512},{1,256,256},{8,96,16}};
    int ok=1;
    for(unsigned i=0;i<sizeof cases/sizeof cases[0];i++){
        int r=check(cases[i].M,cases[i].K,cases[i].N,th);
        printf("M=%d K=%d N=%d th=%d : %s\n",cases[i].M,cases[i].K,cases[i].N,th,r?"OK":"FAIL");
        ok&=r;
    }
    printf("%s\n", ok?"ALL OK":"FAILURES");
    return ok?0:1;
}
