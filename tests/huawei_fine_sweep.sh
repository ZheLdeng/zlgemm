#!/bin/bash
# huawei_fine_sweep.sh — fine-grained i8 GEMM utilisation sweep.
# Parts (MODE=axis|grid|pad|llm|all):
#   axis : per-dimension isolation sweeps (N, M, K) in two co-dim contexts
#   grid : dense 5^3 grid M,K,N in {16,64,256,1024,4096}
#   pad  : padding-boundary probe (N,K around the 16-element tile)
#   llm  : representative shapes pruned from tests/shape.csv (col order m,n,k)
# Per shape: sweep threads, report GOPS-max thread AND the "knee" thread (fewest
# threads within 95% of peak GOPS), plus single-core, with utilisation vs I8_PEAK.
# Timing is e2e: B pre-packed once, A reordered/packed inside every timed call.
#
# Usage:  CORES=0-79 MODE=all bash tests/huawei_fine_sweep.sh
# Knobs:  CORES THREADS RUNS I8_PEAK MODE QUICK OUT FORCE_BUILD

set -u
cd "$(dirname "$0")"
LIB=../lib
B=build/bench_dispatch_i8gemm_sve
NPROC=$(nproc)
CORES="${CORES:-0-$((NPROC-1))}"
RUNS="${RUNS:-4}"
PEAK="${I8_PEAK:-370.9}"
MODE="${MODE:-all}"
THREADS="${THREADS:-1 4 8 16 32 48 64}"
OUT="${OUT:-../results/m8/huawei_fine_sweep.csv}"
if [ "${QUICK:-0}" = 1 ]; then RUNS=2; THREADS="1 8 32 64"; fi
export GOMP_CPU_AFFINITY="$CORES" OMP_PLACES=cores OMP_PROC_BIND=close

echo "############################################################"
echo "# fine sweep  MODE=$MODE QUICK=${QUICK:-0}  CORES=$CORES"
echo "#  THREADS='$THREADS' RUNS=$RUNS PEAK=$PEAK/core  nproc=$NPROC"
echo "############################################################"

make -C "$LIB" sve >/dev/null 2>&1 || { echo "LIB BUILD FAILED"; exit 1; }
if [ ! -x "$B" ] || [ "$LIB/i8gemm_sve.c" -nt "$B" ] || [ "${FORCE_BUILD:-0}" = 1 ]; then
  mkdir -p build
  cc -O3 -Wall -fopenmp -mcpu=native -I"$LIB" -DBENCH_SVE -o "$B" \
    bench_dispatch_types.c $LIB/bf16gemm_sve.c $LIB/bf16gemm_sve.S \
    $LIB/i8gemm_sve.c $LIB/i8gemm_sve.S $LIB/i8gemm_hybrid.S \
    $LIB/i8gemm_pack_a_neon.S -lm || { echo "BENCH BUILD FAILED"; exit 1; }
fi

reps_for(){ local ops=$(( 2*$1*$2*$3 ))
  if   [ $ops -lt 4000000 ];    then echo 400
  elif [ $ops -lt 64000000 ];   then echo 100
  elif [ $ops -lt 1000000000 ]; then echo 20
  else echo 4; fi; }
gops(){ $B i8gemm_sve i8 "$1" "$2" "$3" "$4" 1 "$RUNS" "$5" 2>/dev/null | awk -F, '{print $11}'; }

mkdir -p "$(dirname "$OUT")"
echo "part,M,K,N,best_t,best_gops,best_util,knee_t,knee_gops,knee_util,t1_gops,t1_util" > "$OUT"
printf "%-5s %-16s %6s %9s %6s | %5s %9s %6s | %8s %6s\n" \
  part "M x K x N" "bt" "GOPS" "u%" "kt" "GOPS" "u%" "1c GOPS" "1c%"

# measure PART M K N
measure(){ local part=$1 M=$2 K=$3 N=$4; local reps; reps=$(reps_for $M $K $N)
  local pairs=""; for t in $THREADS; do pairs+="$t:$(gops $M $K $N $reps $t) "; done
  echo "$pairs" | awk -v part="$part" -v m=$M -v k=$K -v n=$N -v pk=$PEAK -v out="$OUT" '{
    bg=0; bt=0; t1=0; nt=split($0,a," ");
    for(i=1;i<=nt;i++){ if(a[i]=="")continue; split(a[i],b,":"); tt[i]=b[1]+0; gg[i]=b[2]+0;
      if(tt[i]==1)t1=gg[i]; if(gg[i]>bg){bg=gg[i];bt=tt[i]} }
    kt=bt; kg=bg;
    for(i=1;i<=nt;i++){ if(gg[i]>=0.95*bg && tt[i]>0 && tt[i]<kt){kt=tt[i];kg=gg[i]} }
    bu=(bt>0)?100*bg/(pk*bt):0; ku=(kt>0)?100*kg/(pk*kt):0; u1=100*t1/pk;
    printf "%-5s %5dx%5dx%-5d %6d %9.1f %5.1f | %5d %9.1f %5.1f | %8.1f %5.1f\n",
      part,m,k,n,bt,bg,bu,kt,kg,ku,t1,u1;
    printf "%s,%d,%d,%d,%d,%.1f,%.1f,%d,%.1f,%.1f,%.1f,%.1f\n",
      part,m,k,n,bt,bg,bu,kt,kg,ku,t1,u1 >> out; }'; }

run_axis(){
  local NL="8 16 24 32 48 64 96 128 192 256 384 512 768 1024 2048 4096"
  local ML="1 2 4 8 16 32 48 64 128 256 512 1024 2048 4096"
  local KL="16 32 64 128 256 512 1024 2048 4096 8192"
  [ "${QUICK:-0}" = 1 ] && { NL="8 16 32 64 128 256 512 1024 4096"; ML="1 4 16 64 256 1024 4096"; KL="16 64 128 512 1024 4096 8192"; }
  echo "-- axis: N sweep (K=1024; ctx M=8,1024) --"
  for mc in 8 1024; do for N in $NL; do measure aN $mc 1024 $N; done; done
  echo "-- axis: M sweep (K=1024; ctx N=256,4096) --"
  for nc in 256 4096; do for M in $ML; do measure aM $M 1024 $nc; done; done
  echo "-- axis: K sweep (N=1024; ctx M=8,1024) --"
  for mc in 8 1024; do for K in $KL; do measure aK $mc $K 1024; done; done
}
run_grid(){
  local G="16 64 256 1024 4096"; [ "${QUICK:-0}" = 1 ] && G="16 256 4096"
  echo "-- grid: M,K,N in {$G} --"
  for M in $G; do for K in $G; do for N in $G; do measure g $M $K $N; done; done; done
}
run_pad(){
  echo "-- pad: N around 16 (M=512,K=1024) --"
  for N in 8 15 16 17 31 32 33 48 63 64; do measure pN 512 1024 $N; done
  echo "-- pad: K around 16 (M=512,N=1024) --"
  for K in 16 17 31 32 48 64; do measure pK 512 $K 1024; done
}
run_llm(){
  echo "-- llm: representative shapes from shape.csv (M K N) --"
  # family A: N=4096, K=512
  for M in 1 8 32 64 256 1024 2048; do measure L $M 512 4096; done
  # family B: N=1024, K=4096
  for M in 1 8 32 64 256 1024 2048; do measure L $M 4096 1024; done
  # specials (M K N)
  for s in "2048 4096 2048" "2048 2048 4096" "2048 1024 8192" "2048 4096 512" \
           "2048 4096 256" "2048 4096 64" "2048 16384 24" "1 4096 32320" \
           "64 512 640" "64 128 512" "2048 128 500" "4 4 4096"; do
    measure L $s; done
}

case "$MODE" in
  axis) run_axis;;
  grid) run_grid;;
  pad)  run_pad;;
  llm)  run_llm;;
  all)  run_axis; run_grid; run_pad; run_llm;;
  *) echo "unknown MODE=$MODE"; exit 2;;
esac

echo "----------------------------------------------------------------------"
echo "csv -> $OUT"
awk -F, 'NR>1{n++; b=$7; u=$12;
  sb+=b; if(bmin==""||b<bmin)bmin=b; if(b>bmax)bmax=b;
  s1+=u; if(u1min==""||u<u1min)u1min=u; if(u>u1max)u1max=u;
  if(u<60){weak[nw++]=sprintf("    %dx%dx%d  1c_util=%.1f%%",$2,$3,$4,u)} }
 END{printf "best-thread util: %.1f..%.1f%% (mean %.1f)\n", bmin,bmax,sb/n;
     printf "single-core util: %.1f..%.1f%% (mean %.1f)  [n=%d, peak=%s/core]\n",u1min,u1max,s1/n,n,"'"$PEAK"'";
     print "single-core WEAK shapes (1c_util<60%):"; for(i=0;i<nw;i++)print weak[i]}' "$OUT"
