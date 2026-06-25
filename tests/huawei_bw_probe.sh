#!/bin/bash
# huawei_bw_probe.sh — decide if a dedicated GEMV/narrow-N kernel (#3/#4) is
# worth writing. M=1 and narrow-N GEMM are memory-bandwidth bound, so low
# "compute utilisation" is expected. This reports the ACHIEVED data bandwidth
# (GB/s = bytes moved / time) and the thread at which it plateaus.
#   bytes = A(M*K) + B(K*N) + C(M*N*4)   (each streamed ~once)
# Compare GB/s to the machine's peak DRAM bandwidth:
#   - achieved ~ peak           -> memory-bound: a new kernel WON'T help (#3/#4 skip)
#   - achieved << peak & flat    -> overhead/instruction-bound: kernel WOULD help
# Also use the knee here + tests/huawei_fine_sweep.sh to calibrate I8_REC_MACS.
#
# Usage: CORES=0-79 PEAK_GBs=<machine peak GB/s> bash tests/huawei_bw_probe.sh

set -u
cd "$(dirname "$0")"
LIB=../lib; B=build/bench_dispatch_i8gemm_sve
CORES="${CORES:-0-$(( $(nproc)-1 ))}"
THREADS="${THREADS:-1 4 8 16 32 48 64}"
RUNS="${RUNS:-4}"
PEAK_GBs="${PEAK_GBs:-0}"   # set to your machine's peak DRAM GB/s for a verdict
export GOMP_CPU_AFFINITY="$CORES" OMP_PLACES=cores OMP_PROC_BIND=close

[ -x "$B" ] || { echo "build $B first (run huawei_fine_sweep.sh once)"; exit 1; }

reps_for(){ local ops=$(( 2*$1*$2*$3 ))
  if [ $ops -lt 4000000 ]; then echo 400; elif [ $ops -lt 64000000 ]; then echo 100;
  elif [ $ops -lt 1000000000 ]; then echo 20; else echo 4; fi; }
gops(){ $B i8gemm_sve i8 "$1" "$2" "$3" "$4" 1 "$RUNS" "$5" 2>/dev/null | awk -F, '{print $11}'; }

echo "############################################################"
echo "# BW probe  CORES=$CORES  PEAK_GBs=${PEAK_GBs:-?}"
echo "#  GB/s = (M*K + K*N + M*N*4) bytes / time;  ops=2*M*K*N"
echo "############################################################"
printf "%-18s %5s %9s %9s %s\n" "M x K x N" "t" "GOPS" "GB/s" "(peak% if PEAK_GBs set)"

probe(){ local M=$1 K=$2 N=$3; local reps; reps=$(reps_for $M $K $N)
  local bytes=$(( M*K + K*N + M*N*4 ))
  echo "-- ${M}x${K}x${N}  bytes=$((bytes/1024))KiB --"
  for t in $THREADS; do
    local g; g=$(gops $M $K $N $reps $t)
    awk -v m=$M -v k=$K -v n=$N -v t=$t -v g="$g" -v by=$bytes -v pk="$PEAK_GBs" 'BEGIN{
      if(g+0<=0){print "   (no result)"; exit}
      tsec=2.0*m*k*n/(g*1e9); gbs=by/tsec/1e9;
      p=(pk+0>0)?sprintf("  %.0f%% of peak",100*gbs/pk):"";
      printf "%18s %5d %9.1f %9.1f%s\n","",t,g,gbs,p }'
  done; }

echo "== M=1 GEMV (decode) =="
probe 1 512 4096      # family A decode
probe 1 4096 1024     # family B decode
probe 1 4096 32320    # huge-N GEMV
echo "== narrow-N =="
probe 2048 4096 64
probe 2048 16384 24
probe 512 1024 16
echo "== large compute-bound reference (for contrast) =="
probe 4096 4096 4096
echo
echo "Verdict guide: if GEMV/narrow-N GB/s plateaus near peak DRAM GB/s -> memory-bound,"
echo "a dedicated GEMV/narrow-N kernel won't help. If GB/s stays far below peak and flat,"
echo "it's overhead-bound and a kernel is worth writing. Send this table back."
