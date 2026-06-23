#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gemm_params.h"
typedef int8_t i8_t; typedef int32_t i32_t;
void i8_pack_B(const i8_t*,i8_t*,int,int);
void i8gemm_k_hybrid(const i8_t*,const i8_t*,i32_t*,i8_t*,const gemm_params_t*);
static int rup(int x,int q){return ((x+q-1)/q)*q;}

static int check(int M,int K,int N){
    int Kr=rup(K,16), Nr=rup(N,16);
    i8_t *A=calloc((size_t)M*Kr,1), *B=calloc((size_t)Kr*Nr,1), *Br=calloc((size_t)Kr*Nr,1);
    i32_t *C=calloc((size_t)M*Nr,sizeof(i32_t)), *R=calloc((size_t)M*Nr,sizeof(i32_t));
    for(int m=0;m<M;m++)for(int k=0;k<K;k++)A[(size_t)m*Kr+k]=(int8_t)((m*7+k*3)%17-8);
    for(int k=0;k<K;k++)for(int n=0;n<N;n++)B[(size_t)k*Nr+n]=(int8_t)((k*5+n)%13-6);
    i8_pack_B(B,Br,Kr,Nr);
    for(int m=0;m<M;m++)for(int n=0;n<Nr;n++){long s=0;for(int k=0;k<Kr;k++)s+=(int)A[(size_t)m*Kr+k]*(int)B[(size_t)k*Nr+n];R[(size_t)m*Nr+n]=(i32_t)s;}
    gemm_params_t p={M,Kr,Nr,Kr,Kr,Nr};
    i8gemm_k_hybrid(A,Br,C,NULL,&p);
    int bad=0; for(size_t i=0;i<(size_t)M*Nr&&bad<5;i++) if(C[i]!=R[i]){printf("  @%zu got %d exp %d\n",i,C[i],R[i]);bad++;}
    free(A);free(B);free(Br);free(C);free(R);
    return bad==0;
}
int main(){
    struct{int M,K,N;}cs[]={{8,64,16},{8,64,64},{64,64,64},{96,96,96},{128,128,128},
                            {5,128,48},{7,80,17},{1,64,32},{13,256,96},{64,32,128},{8,16,16}};
    int ok=1;
    for(unsigned i=0;i<sizeof cs/sizeof cs[0];i++){
        int r=check(cs[i].M,cs[i].K,cs[i].N);
        printf("M=%d K=%d N=%d : %s\n",cs[i].M,cs[i].K,cs[i].N,r?"OK":"FAIL"); ok&=r;
    }
    printf("%s\n",ok?"ALL OK":"FAIL"); return ok?0:1;
}
