#!/bin/bash
# Stable (runs=8) re-measure of the residual loss clusters vs ACL.
export GOMP_CPU_AFFINITY=0-7 OMP_PLACES=cores OMP_PROC_BIND=close
B=build/bench_dispatch_i8gemm_sve; A=build/bench_acl_dispatch
row(){ local m=$1 k=$2 n=$3 t=$4 r=$5
  s=$($B i8gemm_sve i8 $m $k $n $r 2 8 $t|awk -F, '{print $11}')
  a=$($A i8 $m $k $n $r 2 8 $t|awk -F, '{print $10}')
  awk -v m=$m -v k=$k -v n=$n -v t=$t -v s=$s -v a=$a 'BEGIN{
    r=s/a; printf "%5dx%4dx%4d t%d: sve=%8.1f acl=%8.1f ratio=%.3f%s\n",
    m,k,n,t,s,a,r,(r<0.97?"  <ACL":"")}'
}
echo "--- multi-thread K=256/N=256 cubes ---"
for s in "32 256 256" "64 256 256" "128 256 256" "256 256 256"; do row $s 4 80; row $s 8 80; done
echo "--- N=512 cubes @t8 ---"
for s in "64 512 512" "128 512 512" "256 512 512" "512 512 512"; do row $s 8 20; done
echo "--- single-thread small/medium cubes ---"
for s in "16 128 128" "32 256 256" "64 256 256" "128 128 128" "256 256 256"; do row $s 1 200; done
