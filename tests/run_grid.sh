#!/bin/bash
# Quick grid benchmark for SVE bf16/i8 dispatch paths.
# Usage: ./run_grid.sh [dtype] [big]
set -e
cd "$(dirname "$0")"
DTYPE=${1:-both}
BIG=${2:-1}

cc -O3 -Wall -fopenmp -mcpu=native -I../lib -DBENCH_SVE=1 \
  bench_dispatch_types.c \
  ../lib/bf16gemm_sve.c ../lib/bf16gemm_sve.S \
  ../lib/i8gemm_sve.c ../lib/i8gemm_sve.S \
  ../lib/i8gemm_m16n4.c ../lib/i8gemm_m16n4.S \
  ../lib/i8gemm_pack_a_neon.S \
  -o build/bench_dispatch_sve -lm

echo "impl,dtype,cache,M,K,N,threads,batch,kib,reps,perf,pct_base,pct_per_thread"
for shape in "64 512 512" "128 512 1024" "256 512 1024" "512 1024 1024"; do
  set -- $shape
  for t in 1 2 4 8; do
    ./build/bench_dispatch_sve sve $DTYPE $1 $2 $3 30 5 5 $t
  done
done
if [ "$BIG" = "1" ]; then
  for t in 1 2 4 8; do
    ./build/bench_dispatch_sve sve $DTYPE 2048 4096 8192 5 2 3 $t
  done
fi
