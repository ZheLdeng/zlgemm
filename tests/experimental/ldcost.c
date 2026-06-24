#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
void ld_rqb(uint64_t,void*);void ld_1b(uint64_t,void*);void ld_q(uint64_t,void*);
static double ns(){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(){
  void*b=aligned_alloc(64,256);
  uint64_t it=100000000;
  ld_rqb(it/10,b);ld_1b(it/10,b);ld_q(it/10,b);
  double br=1e30,b1=1e30,bq=1e30,t;
  for(int r=0;r<5;r++){t=ns();ld_rqb(it,b);t=ns()-t;if(t<br)br=t;}
  for(int r=0;r<5;r++){t=ns();ld_1b(it,b);t=ns()-t;if(t<b1)b1=t;}
  for(int r=0;r<5;r++){t=ns();ld_q(it,b);t=ns()-t;if(t<bq)bq=t;}
  printf("ld1rqb: %.2f loads/ns\n",8.0*it/br/1e9);
  printf("ld1b  : %.2f loads/ns\n",8.0*it/b1/1e9);
  printf("ldr q : %.2f loads/ns\n",8.0*it/bq/1e9);
  return 0;
}
