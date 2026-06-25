#!/bin/bash
# huawei_gemv_clamp.sh — does the fork-amortisation clamp starve memory-bound
# GEMV (M=1)? Compares the DEFAULT path vs clamp-disabled (MIN_MACS=0) vs forced
# hybrid, across thread counts. If GOPS keeps rising past the default's plateau
# when the clamp is off, the clamp is too tight for GEMV and should be relaxed.
#
# Usage:  CORES=0-79 bash tests/huawei_gemv_clamp.sh
set -u
cd "$(dirname "$0")"
LIB=../lib; B=build/bench_dispatch_i8gemm_sve
CORES="${CORES:-0-$(( $(nproc)-1 ))}"
THREADS="${THREADS:-8 16 32 48 64}"
RUNS="${RUNS:-4}"
export GOMP_CPU_AFFINITY="$CORES" OMP_PLACES=cores OMP_PROC_BIND=close

# build bench if missing (stub the experimental NEON-i8mm m16n4 path)
make -C "$LIB" sve >/dev/null 2>&1 || { echo "LIB BUILD FAILED"; exit 1; }
if [ ! -x "$B" ]; then
  mkdir -p build; STUB=build/_m16n4_stub.c
  cat > "$STUB" <<'EOF'
#include <stdint.h>
void i8gemm_mt_dispatch(const int8_t*, const int8_t*, int32_t*, int, int, int, int);
void i8gemm_mt_dispatch_m16n4(const int8_t* A, const int8_t* B, int32_t* C,
                              int M, int K, int N, int t){ i8gemm_mt_dispatch(A,B,C,M,K,N,t); }
EOF
  cc -O3 -Wall -fopenmp -mcpu=native -I"$LIB" -DBENCH_SVE -o "$B" \
    bench_dispatch_types.c "$STUB" $LIB/bf16gemm_sve.c $LIB/bf16gemm_sve.S \
    $LIB/i8gemm_sve.c $LIB/i8gemm_sve.S $LIB/i8gemm_hybrid.S \
    $LIB/i8gemm_pack_a_neon.S -lm || { echo "BENCH BUILD FAILED"; exit 1; }
fi

g(){ $B i8gemm_sve i8 "$1" "$2" "$3" 200 1 "$RUNS" "$4" 2>/dev/null | awk -F, '{print $11}'; }

echo "############################################################"
echo "# GEMV clamp test  CORES=$CORES THREADS='$THREADS' RUNS=$RUNS"
echo "#  cols: default | MIN_MACS=0 (no clamp) | HYBRID=1   (GOPS, 2-op)"
echo "############################################################"
row(){ local M=$1 K=$2 N=$3
  echo "-- ${M}x${K}x${N} --"
  for t in $THREADS; do
    local d c h
    d=$(g $M $K $N $t)
    c=$(I8_SVE_MIN_MACS=0 g $M $K $N $t)
    h=$(I8_SVE_HYBRID=1 g $M $K $N $t)
    awk -v t=$t -v d="$d" -v c="$c" -v h="$h" 'BEGIN{
      printf "  t%-3d default=%-8.1f noclamp=%-8.1f hybrid=%-8.1f  noclamp/default=%.2fx\n",
        t,d,c,h,(d>0?c/d:0)}'
  done; }

row 1 4096 1024       # the suspected starved GEMV
row 1 512  4096       # already hybrid (reference)
row 1 4096 4096       # large-N GEMV
row 1 11008 4096      # LLM down-proj GEMV
row 1 4096 11008      # LLM up-proj GEMV

echo "----------------------------------------------------------------------"
echo "Verdict: if noclamp/hybrid GOPS keeps rising past the default plateau"
echo "(noclamp/default >> 1 at high t), the clamp starves GEMV -> I relax it."
echo "If all three columns are ~flat & equal, the clamp is fine; GEMV done."
