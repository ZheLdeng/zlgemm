#!/bin/bash
# huawei_shape_sweep.sh — 27-combo (M,N,K in {small,med,large}) utilisation sweep.
# For each shape: sweep thread counts, pick the GOPS-max thread, report GOPS +
# utilisation at that thread and at 1 core. Timing is e2e: B is pre-packed once
# (weights), A is reordered/packed inside every timed call (activation) — exactly
# the "B packed, A not pre-packed" inference case.
#
# Utilisation = perf / (I8_PEAK * threads). I8_PEAK defaults to the Huawei
# measured single-core SVE-i8 ceiling 370.9 GOPS (2-op). Override if needed.
#
# Usage:  CORES=0-79 bash tests/huawei_shape_sweep.sh
# Knobs:  CORES, THREADS, RUNS, I8_PEAK, SMALL/MED/LARGE (the three size points),
#         FORCE_BUILD, OUT (csv path).

set -u
cd "$(dirname "$0")"
LIB=../lib
B=build/bench_dispatch_i8gemm_sve

NPROC=$(nproc)
CORES="${CORES:-0-$((NPROC-1))}"
THREADS="${THREADS:-1 4 8 16 32 48 64 79}"
RUNS="${RUNS:-4}"
PEAK="${I8_PEAK:-370.9}"
SMALL="${SMALL:-64}"; MED="${MED:-512}"; LARGE="${LARGE:-2048}"
OUT="${OUT:-../results/m8/huawei_shape_sweep.csv}"
export GOMP_CPU_AFFINITY="$CORES" OMP_PLACES=cores OMP_PROC_BIND=close

echo "############################################################"
echo "# 27-combo utilisation sweep (e2e: B prepacked, A packed per-call)"
echo "#  CORES=$CORES THREADS='$THREADS' RUNS=$RUNS PEAK=$PEAK GOPS/core"
echo "#  sizes: small=$SMALL med=$MED large=$LARGE   nproc=$NPROC"
echo "############################################################"

# ---- build (stub out the experimental NEON-i8mm m16n4 path) ----
make -C "$LIB" sve >/dev/null 2>&1 || { echo "LIB BUILD FAILED"; exit 1; }
if [ ! -x "$B" ] || [ "${FORCE_BUILD:-0}" = 1 ]; then
  mkdir -p build
  STUB=build/_m16n4_stub.c
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

reps_for(){ local ops=$(( 2*$1*$2*$3 ))
  if   [ $ops -lt 4000000 ];   then echo 400
  elif [ $ops -lt 64000000 ];  then echo 100
  elif [ $ops -lt 1000000000 ];then echo 20
  else echo 4; fi; }
gops(){ $B i8gemm_sve i8 "$1" "$2" "$3" "$4" 1 "$RUNS" "$5" 2>/dev/null | awk -F, '{print $11}'; }

mkdir -p "$(dirname "$OUT")"
echo "M,K,N,best_t,best_gops,best_util_pct,t1_gops,t1_util_pct" > "$OUT"
printf "\n%-16s %7s %10s %8s | %8s %7s\n" "M x K x N" "best_t" "GOPS" "util%" "1c GOPS" "1c%"
echo "----------------------------------------------------------------------"

SIZES="$SMALL $MED $LARGE"
for M in $SIZES; do for K in $SIZES; do for N in $SIZES; do
  reps=$(reps_for $M $K $N)
  t1=$(gops $M $K $N $reps 1)
  best_t=1; best_g="$t1"
  for t in $THREADS; do
    [ "$t" = 1 ] && continue
    g=$(gops $M $K $N $reps $t)
    best=$(awk -v a="$g" -v b="$best_g" 'BEGIN{print (a>b)?1:0}')
    [ "$best" = 1 ] && { best_g="$g"; best_t="$t"; }
  done
  awk -v m=$M -v k=$K -v n=$N -v bt=$best_t -v bg="$best_g" -v t1="$t1" -v pk="$PEAK" -v out="$OUT" 'BEGIN{
    bu=100*bg/(pk*bt); u1=100*t1/pk;
    printf "%5dx%5dx%-5d %7d %10.1f %7.1f%% | %8.1f %6.1f%%\n", m,k,n,bt,bg,bu,t1,u1;
    printf "%d,%d,%d,%d,%.1f,%.1f,%.1f,%.1f\n", m,k,n,bt,bg,bu,t1,u1 >> out;
  }'
done; done; done

echo "----------------------------------------------------------------------"
echo "csv -> $OUT"
echo "== utilisation summary (best-thread) =="
awk -F, 'NR>1{u=$6; n++; s+=u; if(min==""||u<min)min=u; if(u>max)max=u;
  if($1<=64&&$2<=64&&$3<=64)sc="all-small";
  } END{printf "  best-thread util: min=%.1f%% max=%.1f%% mean=%.1f%% (n=%d, peak=%s/core)\n",min,max,s/n,n,"'"$PEAK"'"}' "$OUT"
awk -F, 'NR>1{u=$8;n++;s+=u;if(min==""||u<min)min=u;if(u>max)max=u}END{
  printf "  single-core util: min=%.1f%% max=%.1f%% mean=%.1f%%\n",min,max,s/n}' "$OUT"
echo "(util = GOPS / (peak * threads); e2e includes A pack, B pre-packed.)"
