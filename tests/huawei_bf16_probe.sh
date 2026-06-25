#!/bin/bash
# huawei_bf16_probe.sh — does bf16 have the same scheduling weak spots i8 had?
# bf16 uses a DIFFERENT scheduler than i8 (a (m_unit x n_tile) 2D *collapse*,
# not i8's tall-pm/thin-pn pick_grid), so i8's narrow-N grid pathology may not
# exist here. This probes whether any forced split mode beats the default on the
# shapes that were weak for i8: narrow-N (large M/K), small-M/GEMV, medium cubes.
# Compares default vs BF16_SVE_SPLIT={m,n,2d} at a few thread counts.
#   BEST=default with small +% -> bf16 scheduler already optimal (no transfer).
#   BEST=split* with big +%     -> a cheap bf16 scheduling fix exists; send table.
#
# Usage: CORES=0-79 bash tests/huawei_bf16_probe.sh
set -u
cd "$(dirname "$0")"
LIB=../lib; B=build/bench_dispatch_i8gemm_sve
CORES="${CORES:-0-$(( $(nproc)-1 ))}"
RUNS="${RUNS:-4}"
export GOMP_CPU_AFFINITY="$CORES" OMP_PLACES=cores OMP_PROC_BIND=close

make -C "$LIB" sve >/dev/null 2>&1 || { echo "LIB BUILD FAILED"; exit 1; }
if [ ! -x "$B" ] || [ "$LIB/bf16gemm_sve.c" -nt "$B" ] || [ "${FORCE_BUILD:-0}" = 1 ]; then
  echo "building $B ..."; mkdir -p build
  cc -O3 -Wall -fopenmp -mcpu=native -I"$LIB" -DBENCH_SVE -o "$B" \
    bench_dispatch_types.c $LIB/bf16gemm_sve.c $LIB/bf16gemm_sve.S \
    $LIB/i8gemm_sve.c $LIB/i8gemm_sve.S $LIB/i8gemm_hybrid.S \
    $LIB/i8gemm_pack_a_neon.S -lm || { echo "BENCH BUILD FAILED"; exit 1; }
fi

g(){ env "$1" $B i8gemm_sve bf16 "$2" "$3" "$4" "$5" 1 "$RUNS" "$6" 2>/dev/null | awk -F, '{print $11}'; }

echo "############################################################"
echo "# bf16 split-mode probe  CORES=$CORES RUNS=$RUNS  (GFLOPS, 2-op)"
echo "############################################################"
probe(){ local M=$1 K=$2 N=$3 reps=$4; shift 4
  echo "-- ${M}x${K}x${N} --"
  for t in "$@"; do
    local d sm sn s2
    d=$(g  X=0                 $M $K $N $reps $t)
    sm=$(g BF16_SVE_SPLIT=m    $M $K $N $reps $t)
    sn=$(g BF16_SVE_SPLIT=n    $M $K $N $reps $t)
    s2=$(g BF16_SVE_SPLIT=2d   $M $K $N $reps $t)
    awk -v t=$t -v d="$d" -v sm="$sm" -v sn="$sn" -v s2="$s2" 'BEGIN{
      best=d; nm="default";
      if(sm>best){best=sm;nm="m"} if(sn>best){best=sn;nm="n"} if(s2>best){best=s2;nm="2d"}
      printf "  t%-3d default=%-8.1f m=%-8.1f n=%-8.1f 2d=%-8.1f  BEST=%s(%.0f, %+.0f%% vs def)\n",
        t,d,sm,sn,s2,nm,best,(d>0?100*(best-d)/d:0)}'
  done; }

echo "== narrow-N large-M-K (i8 had a 2D pathology here) =="
probe 2048 4096 64  8  32 48 64
probe 2048 16384 24 8  32 48 64
echo "== small-M / GEMV (i8 clamp starved these) =="
probe 8  4096 1024  20 16 32 64
probe 1  4096 4096  40 16 32 64
echo "== medium cube (i8 over-parallelised) =="
probe 256 256 256   100 16 32 64
echo "== large (no-regression reference) =="
probe 2048 4096 2048 4  32 64
echo "----------------------------------------------------------------------"
echo "If BEST=default (or +% small) everywhere -> bf16 scheduler already good."
echo "If a split mode wins big on narrow-N/small-M -> send table, I add the fix."
