#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
void scatter_store(uint64_t,int32_t*); void contig_store(uint64_t,int32_t*);
static double ns(){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(){
  int32_t *b=aligned_alloc(64,4096);
  uint64_t it=50000000;
  scatter_store(it/10,b); contig_store(it/10,b);
  double s=1e9; for(int r=0;r<5;r++){double t=ns();scatter_store(it,b);double d=ns()-t;if(d<s)s=d;}
  double c=1e9; for(int r=0;r<5;r++){double t=ns();contig_store(it,b);double d=ns()-t;if(d<c)c=d;}
  printf("scatter: %.2f st/ns  contig: %.2f st/ns  ratio=%.2fx\n",
         8.0*it/s/1e9, 8.0*it/c/1e9, s/c);
  printf("per 8-store tile: scatter=%.1f ns, contig=%.1f ns\n", s/it, c/it);
  return 0;
}
