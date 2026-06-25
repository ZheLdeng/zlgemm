#!/bin/bash
# huawei_blocking_probe.sh — is there cache-blocking headroom for the
# bandwidth-bound medium / narrow-N shapes, and is it cheap (a scheduling-mode
# switch) or does it need real K-blocking?
# For each shape, compare the current default scheduling vs forced variants at a
# few thread counts:
#   PACK2D=0   : disable the packed 2D thread grid (pure M/N split)
#   SPLIT=n    : force N-split          SPLIT=m : force M-split
#   HYBMT=1/2/4: hybrid N-split / 2D-collapse / 2D-grid (for hybrid-routed shapes)
# If the default already wins -> 2D grid captures the benefit, no cheap headroom.
# If a forced variant clearly beats default -> cheap scheduling fix.
# If nothing beats default and util stays low -> needs real K/2D cache-blocking.
#
# Usage: CORES=0-79 bash tests/huawei_blocking_probe.sh
set -u
cd "$(dirname "$0")"
LIB=../lib; B=build/bench_dispatch_i8gemm_sve
CORES="${CORES:-0-$(( $(nproc)-1 ))}"
RUNS="${RUNS:-4}"
export GOMP_CPU_AFFINITY="$CORES" OMP_PLACES=cores OMP_PROC_BIND=close
# (re)build the bench if missing OR older than the lib source (so a git pull is
# picked up), stubbing the experimental NEON-i8mm m16n4 path.
if [ ! -x "$B" ] || [ "$LIB/i8gemm_sve.c" -nt "$B" ] || [ "${FORCE_BUILD:-0}" = 1 ]; then
  echo "building $B ..."; mkdir -p build; STUB=build/_m16n4_stub.c
  cat > "$STUB" <<'EOF'
#include <stdint.h>
void i8gemm_mt_dispatch(const int8_t*, const int8_t*, int32_t*, int, int, int, int);
void i8gemm_mt_dispatch_m16n4(const int8_t* A, const int8_t* B, int32_t* C,
                              int M, int K, int N, int t){ i8gemm_mt_dispatch(A,B,C,M,K,N,t); }
EOF
  make -C "$LIB" sve >/dev/null 2>&1
  cc -O3 -Wall -fopenmp -mcpu=native -I"$LIB" -DBENCH_SVE -o "$B" \
    bench_dispatch_types.c "$STUB" $LIB/bf16gemm_sve.c $LIB/bf16gemm_sve.S \
    $LIB/i8gemm_sve.c $LIB/i8gemm_sve.S $LIB/i8gemm_hybrid.S \
    $LIB/i8gemm_pack_a_neon.S -lm || { echo "BENCH BUILD FAILED"; exit 1; }
fi

g(){ env "$1" $B i8gemm_sve i8 "$2" "$3" "$4" "$5" 1 "$RUNS" "$6" 2>/dev/null | awk -F, '{print $11}'; }

echo "############################################################"
echo "# blocking headroom probe  CORES=$CORES RUNS=$RUNS"
echo "############################################################"
# shape  reps  thread-list
probe(){ local M=$1 K=$2 N=$3 reps=$4; shift 4; local TS="$*"
  echo "-- ${M}x${K}x${N} --"
  for t in $TS; do
    local d p0 sn sm
    d=$(g  X=0          $M $K $N $reps $t)
    p0=$(g I8_SVE_PACK2D=0 $M $K $N $reps $t)
    sn=$(g I8_SVE_SPLIT=n  $M $K $N $reps $t)
    sm=$(g I8_SVE_SPLIT=m  $M $K $N $reps $t)
    awk -v t=$t -v d="$d" -v p0="$p0" -v sn="$sn" -v sm="$sm" 'BEGIN{
      best=d; nm="default";
      if(p0>best){best=p0;nm="pack2d=0"} if(sn>best){best=sn;nm="split=n"} if(sm>best){best=sm;nm="split=m"}
      printf "  t%-3d default=%-8.1f pack2d0=%-8.1f splitN=%-8.1f splitM=%-8.1f  BEST=%s(%.0f, %+.0f%% vs def)\n",
        t,d,p0,sn,sm,nm,best,(d>0?100*(best-d)/d:0)}'
  done; }

echo "== medium over-parallelisers (knee << 64) =="
probe 32  4096 1024 20  16 32 64
probe 64  4096 1024 20  16 32 64
probe 256 256  256  100 16 32 64
echo "== narrow-N (low multi-thread util, B re-streaming) =="
probe 2048 4096 64   8   32 48 64
probe 2048 16384 24  8   32 48 64
probe 1024 4096 256  8   32 48 64
echo "----------------------------------------------------------------------"
echo "Read: BEST=default everywhere -> no cheap headroom (needs real K-blocking)."
echo "BEST=split*/pack2d0 with big +% -> cheap scheduling fix available; send table."
