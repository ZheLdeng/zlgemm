#!/bin/bash
# huawei_validate.sh — one-shot build + correctness + scheduling validation for the
# Huawei 80-core box. Validates (1) the t79 prime-thread collapse fix, (2) the gap#1
# small-M scaling fix, and (3) no-regression on large shapes (packed path untouched).
# Self-contained: needs NO ACL/KleidiAI.
#
# Usage:
#   CORES=80-159 bash tests/huawei_validate.sh
#
# Env knobs:
#   CORES        cpu set to bind (default: 0..nproc-1). SET THIS to your 80-core node!
#   THREADS      thread counts to sweep (default: "1 2 4 8 16 32 64 79 80")
#   RUNS         best-of-runs per point (default 4)
#   FORCE_BUILD  =1 to rebuild the bench binaries even if present
#
# bench_dispatch output: CSV, perf(GOPS, 2-op) is column 11.

set -u
cd "$(dirname "$0")"                       # tests/
LIB=../lib
B=build/bench_dispatch_i8gemm_sve

NPROC=$(nproc)
CORES="${CORES:-0-$((NPROC-1))}"
THREADS="${THREADS:-1 2 4 8 16 32 64 79 80}"
RUNS="${RUNS:-4}"
export GOMP_CPU_AFFINITY="$CORES" OMP_PLACES=cores OMP_PROC_BIND=close

echo "############################################################"
echo "# Huawei i8gemm scheduling validation"
echo "#   CORES=$CORES  THREADS='$THREADS'  RUNS=$RUNS  nproc=$NPROC"
echo "############################################################"

# ---------- 1. build ----------
echo; echo "== [1/4] build lib + bench =="
make -C "$LIB" sve >/dev/null 2>&1 || { echo "LIB BUILD FAILED"; exit 1; }
if [ ! -x "$B" ] || [ "${FORCE_BUILD:-0}" = 1 ]; then
  echo "  compiling $B ..."
  mkdir -p build
  # The experimental NEON-i8mm m16n4 path (i8gemm_m16n4.{c,S}) is irrelevant to
  # scheduling validation and its AdvSIMD `smmla v.4s,v.16b` needs the i8mm
  # feature enabled for C intrinsics (which `-mcpu=native` may not turn on for
  # AdvSIMD on every toolchain). bench_dispatch_types.c only *references* the
  # symbol i8gemm_mt_dispatch_m16n4, so we satisfy it with a tiny stub and skip
  # compiling the NEON-i8mm code entirely -> builds wherever `make sve` does.
  STUB=build/_m16n4_stub.c
  cat > "$STUB" <<'EOF'
#include <stdint.h>
void i8gemm_mt_dispatch(const int8_t*, const int8_t*, int32_t*, int, int, int, int);
/* Never called in validation (impl != "m16n4"); forward to the SVE dispatch. */
void i8gemm_mt_dispatch_m16n4(const int8_t* A, const int8_t* B, int32_t* C,
                              int M, int K, int N, int t) {
    i8gemm_mt_dispatch(A, B, C, M, K, N, t);
}
EOF
  cc -O3 -Wall -fopenmp -mcpu=native -I"$LIB" -DBENCH_SVE -o "$B" \
    bench_dispatch_types.c "$STUB" $LIB/bf16gemm_sve.c $LIB/bf16gemm_sve.S \
    $LIB/i8gemm_sve.c $LIB/i8gemm_sve.S $LIB/i8gemm_hybrid.S \
    $LIB/i8gemm_pack_a_neon.S -lm || { echo "BENCH BUILD FAILED"; exit 1; }
fi
echo "  ok."

# ---------- 2. correctness ----------
echo; echo "== [2/4] correctness (must pass before trusting perf) =="
make -C . test-sve 2>/dev/null    | grep -iE "correct|OK|pass|fail" || echo "  test-sve: NO RESULT (check build)"
make -C . test-msplit 2>/dev/null | grep -iE "correct|OK|pass|fail" || echo "  test-msplit: NO RESULT (check build)"

# ---------- helpers ----------
# perf for one (shape, threads): echoes GOPS
gops(){ $B i8gemm_sve i8 "$1" "$2" "$3" "$4" 1 "$RUNS" "$5" 2>/dev/null | awk -F, '{print $11}'; }
# sweep a shape over $THREADS, print "  M x K x N : t1=.. t2=.. ..."
sweep(){ local m=$1 k=$2 n=$3 reps=$4; printf "  %5dx%-5dx%-5d:" "$m" "$k" "$n"
  for t in $THREADS; do printf " t%s=%s" "$t" "$(gops $m $k $n $reps $t)"; done; echo; }
# t79 recovery verdict: ratio of t79 to max(t64,t80)
recov(){ local m=$1 k=$2 n=$3 reps=$4
  local t64 t79 t80; t64=$(gops $m $k $n $reps 64); t79=$(gops $m $k $n $reps 79); t80=$(gops $m $k $n $reps 80)
  awk -v m=$m -v k=$k -v n=$n -v a=$t64 -v b=$t79 -v c=$t80 'BEGIN{
    ref=(c>a?c:a); r=(ref>0?b/ref:0);
    printf "  %dx%dx%d: t64=%.0f t79=%.0f t80=%.0f  t79/max(t64,t80)=%.2f  %s\n",
      m,k,n,a,b,c,r,(r>=0.85?"OK (collapse fixed)":(r>=0.7?"PARTIAL":"STILL COLLAPSING <-- investigate"))}'; }

# ---------- 3. t79 collapse fix ----------
echo; echo "== [3/4] t79 prime-thread collapse (was ~42% of t64/t80; want >=0.85) =="
recov 2048 4096 2048 4
recov 2048 4096 1024 4
recov 1024 4096 1024 6

# ---------- 4. gap#1 small-M scaling + large-shape no-regression ----------
echo; echo "== [4/4a] gap#1 small-M scaling (should rise monotonically, not flatten after t8) =="
sweep 8   512  512  50
sweep 16  512  512  50
sweep 8   1024 1024 20
sweep 16  1024 1024 20

echo; echo "== [4/4b] large-shape NO-REGRESSION (packed path, 512Ki clamp preserved) =="
sweep 2048 2048 2048 4
sweep 2048 4096 8192 3
sweep 512  1024 1024 8

echo; echo "############################################################"
echo "# Done. Send me: the [3/4] verdicts, the [4/4a] sweeps, and the"
echo "# [4/4b] numbers vs your historical m8_results_new.xlsx (t64/t79/t80)."
echo "############################################################"
