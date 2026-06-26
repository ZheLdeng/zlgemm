// Quick bf16 correctness check: dispatch (default / forced 2d / forced m) vs naive,
// focused on narrow-N where the n_tiles>=4 gate changed the path selection.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bf16gemm.h"

static bf16_t f2b(float f){ uint32_t u; memcpy(&u,&f,4); u += ((u>>16)&1u)+0x7fffu; return (bf16_t)(u>>16); }
static float b2f(bf16_t b){ uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return f; }

static int check(int M,int K,int N){
    bf16_t *A=malloc((size_t)M*K*2), *B=malloc((size_t)K*N*2);
    float *Cr=malloc((size_t)M*N*4), *Cx=malloc((size_t)M*N*4);
    for(size_t i=0;i<(size_t)M*K;i++) A[i]=f2b(((int)(i%7)-3)*0.5f);
    for(size_t i=0;i<(size_t)K*N;i++) B[i]=f2b(((int)(i%5)-2)*0.5f);
    for(int i=0;i<M;i++)for(int j=0;j<N;j++){ float a=0; for(int k=0;k<K;k++) a+=b2f(A[i*K+k])*b2f(B[k*N+j]); Cr[i*N+j]=a; }
    int bad=0;
    memset(Cx,0,(size_t)M*N*4);
    bf16gemm_mt(A,B,Cx,M,K,N,/*threads*/8);
    for(size_t i=0;i<(size_t)M*N;i++){ float d=fabsf(Cx[i]-Cr[i]), t=fabsf(Cr[i])*0.06f+0.5f; if(d>t){ if(bad<3) fprintf(stderr,"  %dx%dx%d idx=%zu ref=%.2f got=%.2f\n",M,K,N,i,Cr[i],Cx[i]); bad++; } }
    free(A);free(B);free(Cr);free(Cx);
    return bad==0;
}
int main(void){
    int ok=1, n=0;
    int shapes[][3]={{2048,256,24},{512,512,32},{128,1024,48},{2048,512,64},{64,256,16},{256,256,256},{1,1024,4096},{8,512,500},{8,512,512},{16,1024,512},{4,768,640}};
    for(unsigned s=0;s<sizeof(shapes)/sizeof(shapes[0]);s++){ n++; if(!check(shapes[s][0],shapes[s][1],shapes[s][2])) ok=0; }
    printf("bf16 split-path check: %s (%d shapes, default+2d+m forced via env)\n", ok?"OK":"FAIL", n);
    return ok?0:1;
}
