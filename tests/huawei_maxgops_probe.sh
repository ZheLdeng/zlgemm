#!/bin/bash
# huawei_maxgops_probe.sh — MAX-GOPS audit (cores don't matter, only throughput).
# Question: does the thread clamp ever cut BELOW the GOPS-max point (losing perf)?
#   i8:   default (clamp on, 512Ki macs-floor)  vs  I8_SVE_MIN_MACS=0 (work_units cap only)
#   bf16: default (clamp OFF = all threads)      vs  BF16_SVE_CLAMP_THREADS=1 (clamp on)
# For each shape at high thread counts, report both and the max.
#   clampoff > default (i8)  -> clamp is LEAVING GOPS on the table -> relax for max GOPS.
#   clampon  > default (bf16)-> using all threads COLLAPSES -> bf16 needs a clamp for max GOPS.
#   equal                    -> current default already gives max GOPS.
#
# Usage: CORES=0-79 bash tests/huawei_maxgops_probe.sh
set -u
cd "$(dirname "$0")"
LIB=../lib; B=build/bench_dispatch_i8gemm_sve
CORES="${CORES:-0-$(( $(nproc)-1 ))}"
THREADS="${THREADS:-64 79 80}"
RUNS="${RUNS:-4}"
export GOMP_CPU_AFFINITY="$CORES" OMP_PLACES=cores OMP_PROC_BIND=close

make -C "$LIB" sve >/dev/null 2>&1 || { echo "LIB BUILD FAILED"; exit 1; }
if [ ! -x "$B" ] || [ "$LIB/i8gemm_sve.c" -nt "$B" ] || [ "$LIB/bf16gemm_sve.c" -nt "$B" ] || [ "${FORCE_BUILD:-0}" = 1 ]; then
  echo "building $B ..."; mkdir -p build
  cc -O3 -Wall -fopenmp -mcpu=native -I"$LIB" -DBENCH_SVE -o "$B" \
    bench_dispatch_types.c $LIB/bf16gemm_sve.c $LIB/bf16gemm_sve.S \
    $LIB/i8gemm_sve.c $LIB/i8gemm_sve.S $LIB/i8gemm_hybrid.S \
    $LIB/i8gemm_pack_a_neon.S -lm || { echo "BENCH BUILD FAILED"; exit 1; }
fi

gv(){ env "$1" $B i8gemm_sve "$2" "$3" "$4" "$5" "$6" 1 "$RUNS" "$7" 2>/dev/null | awk -F, '{print $11}'; }

# dtype default_env clampvariant_env  shapes...
audit(){ local dt=$1 defenv=$2 varenv=$3 varname=$4; shift 4
  echo "== $dt :  default  vs  $varname =="
  for s in "$@"; do set -- $s; local M=$1 K=$2 N=$3
    local reps; reps=$(( 2*M*K*N<1000000000 ? 20 : 4 ))
    for t in $THREADS; do
      local d v
      d=$(gv "$defenv" $dt $M $K $N $reps $t)
      v=$(gv "$varenv" $dt $M $K $N $reps $t)
      awk -v dt=$dt -v m=$M -v k=$K -v n=$N -v t=$t -v d="$d" -v v="$v" -v vn="$varname" 'BEGIN{
        hi=(v>d)?v:d; who=(v>d*1.03)?(vn" WINS"):((d>v*1.03)?"default wins":"~equal");
        printf "  %5dx%5dx%-5d t%-2d  default=%-8.1f %s=%-8.1f  max=%-8.1f  %s\n",
          m,k,n,t,d,vn,v,hi,who}'
    done
  done; }

# i8: does disabling the macs-floor (max parallelism) beat the default clamp?
audit i8 "X=0" "I8_SVE_MIN_MACS=0" "noclamp" \
  "64 512 256" "128 256 256" "256 256 256" "8 512 512" "64 4096 1024" \
  "512 512 512" "2048 4096 2048" "256 1024 1024" "1024 256 256"
echo
# bf16: does enabling a clamp ever beat using all threads?
audit bf16 "X=0" "BF16_SVE_CLAMP_THREADS=1" "clampon" \
  "64 512 256" "128 256 256" "256 256 256" "8 512 512" "64 4096 1024" \
  "512 512 512" "2048 4096 2048" "256 1024 1024" "1024 256 256"
echo "----------------------------------------------------------------------"
echo "Read: any 'noclamp WINS' (i8) => clamp cuts max GOPS, relax it."
echo "      any 'clampon WINS' (bf16) => all-threads collapses, bf16 needs clamp."
echo "      mostly '~equal' => current defaults already give max GOPS."
