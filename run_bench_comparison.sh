#!/bin/bash
#===============================================================================
# i8gemm_k_ldc vs i8gemm_k_ldc_lda vs i8gemm_k_pack — correctness & performance
# Date: 2026-06-02
#===============================================================================
set -e

ARCH="-march=armv8.6-a+i8mm"
CFLAGS="-O2 -Wall"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
LOGFILE="bench_lda_comparison.log"

exec > >(tee "$LOGFILE") 2>&1

echo "================================================================================"
echo "  i8gemm Kernel Comparison: ldc / lda / pack (internal A+B reorder)"
echo "  Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  CPU:  $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo Unknown)"
echo "  Arch: $(uname -m) / $(sw_vers -productVersion 2>/dev/null || echo macOS)"
echo "================================================================================"

# ==============================================================================
# 1. Build correctness test (lda kernel)
# ==============================================================================
echo ""
echo "── [1/4] Building correctness test (lda kernel) ──"
cc -o calculate_lda_test calculatei8mm.c i8gemm_k_ldc_lda.S $ARCH $CFLAGS
echo "  -> calculate_lda_test built OK"

# ==============================================================================
# 1b. Build correctness test (pack kernel)
# ==============================================================================
echo ""
echo "── [1b] Building correctness test (pack kernel) ──"
cc -o calculate_pack_test calculatei8mm_pack.c i8gemm_k_pack.S $ARCH $CFLAGS
echo "  -> calculate_pack_test built OK"

# ==============================================================================
# 2. Correctness verification — random matrices, many sizes
# ==============================================================================
echo ""
echo "── [2/4] Correctness sweep (random matrices) ──"

FAILED=0; TOTAL=0
for M in 8 16 24 32 40 48 56 64; do
  for K in 16 32 48 64 80; do
    for N in 8 16 24 32; do
      TOTAL=$((TOTAL + 1))
      RESULT=$(./calculate_lda_test $M $K $N 2>&1 | tail -1)
      if [ "$RESULT" != "successfully!" ]; then
        echo "  FAIL (lda): M=$M K=$K N=$N"
        FAILED=$((FAILED + 1))
      fi
      RESULT2=$(./calculate_pack_test $M $K $N 2>&1 | tail -1)
      if [ "$RESULT2" != "successfully!" ]; then
        echo "  FAIL (pack): M=$M K=$K N=$N"
        FAILED=$((FAILED + 1))
      fi
    done
  done
done

echo "  Correctness: $((TOTAL * 2 - FAILED)) / $((TOTAL * 2)) passed"
if [ $FAILED -gt 0 ]; then
  echo "  ERROR: $FAILED failures — aborting!"
  exit 1
fi
echo "  All correctness checks passed."

# ==============================================================================
# 3. Build performance benchmark binary
# ==============================================================================
echo ""
echo "── [3/4] Building benchmark binary ──"

# Rename global symbols so all three kernels link together
sed 's/_i8gemm_k/_i8gemm_k_ldc/g'  i8gemm_k_ldc.S      > /tmp/_ldc.S
sed 's/_i8gemm_k/_i8gemm_k_lda/g'  i8gemm_k_ldc_lda.S  > /tmp/_lda.S
sed 's/_i8gemm_k_pack/_i8gemm_k_pack/g' i8gemm_k_pack.S > /tmp/_pack.S
cc -c /tmp/_ldc.S  -o /tmp/_ldc.o  $ARCH $CFLAGS
cc -c /tmp/_lda.S  -o /tmp/_lda.o  $ARCH $CFLAGS
cc -c /tmp/_pack.S -o /tmp/_pack.o $ARCH $CFLAGS
cc -o bench_lda_cmp bench_lda.c /tmp/_ldc.o /tmp/_lda.o /tmp/_pack.o $ARCH $CFLAGS
rm -f /tmp/_ldc.S /tmp/_lda.S /tmp/_pack.S /tmp/_ldc.o /tmp/_lda.o /tmp/_pack.o
echo "  -> bench_lda_cmp built OK"

# ==============================================================================
# 4. Performance benchmarks
# ==============================================================================
echo ""
echo "── [4/4] Performance comparison ──"
echo ""

run_bench() {
  local m=$1 k=$2 n=$3 loops=$4
  ./bench_lda_cmp $m $k $n $loops 2>/dev/null
}

divider() {
  printf "  %5s %5s %5s  %10s  %10s  %10s  %8s  %8s  %8s  %7s  %7s\n" \
         "-----" "-----" "-----" "----------" "----------" "----------" \
         "--------" "--------" "--------" "-------" "-------"
}

header() {
  printf "  %5s %5s %5s  %10s  %10s  %10s  %8s  %8s  %8s  %7s  %7s\n" \
         "M" "K" "N" "t_ldc(us)" "t_lda(us)" "t_pack(us)" \
         "GF_ldc" "GF_lda" "GF_pack" "d_lda%" "d_pack%"
  divider
}

# ---- 4a. Varying N (M, K fixed) ----
echo "  --- Varying N (M=64 K=128) ---"
header
for N in 8 16 32 64 128 256; do
  run_bench 64 128 $N 200
done
echo ""

# ---- 4b. Varying K (M, N fixed) ----
echo "  --- Varying K (M=64 N=64) ---"
header
for K in 16 32 64 128 256 512; do
  run_bench 64 $K 64 200
done
echo ""

# ---- 4c. Varying M (K, N fixed) ----
echo "  --- Varying M (K=128 N=64) ---"
header
for M in 8 16 32 64 128 256; do
  run_bench $M 128 64 200
done
echo ""

# ---- 4d. Full-size sweep ----
echo "  --- Full-size sweep ---"
header
for M in 64 128 256; do
  for K in 128 256 512; do
    for N in 64 128 256; do
      LOOPS=100; [ $M -ge 256 ] && LOOPS=50
      run_bench $M $K $N $LOOPS
    done
  done
done
echo ""

# ---- 4e. Stabilized peak (repeated runs) ----
echo "  --- Stabilized peak (3 runs each) ---"
header
for size in "128 256 256 100" "256 256 256 100" "256 512 128 100" "512 256 128 50"; do
  echo "  # $size"
  read M K N LOOPS <<< "$size"
  for i in 1 2 3; do
    run_bench $M $K $N $LOOPS
  done
  divider
done

# ---- 4f. Large LLM-like shapes ----
echo ""
echo "  --- LLM-like shapes ---"
header
# Decode-like: small M, large K & N
for M in 8 16 32; do
  for K in 4096 8192; do
    for N in 4096 8192; do
      run_bench $M $K $N 5
    done
  done
done

# Prefill-like: moderate M, large K & N
for M in 256 512; do
  K=4096; N=4096
  run_bench $M $K $N 3
done

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "================================================================================"
echo "  Benchmark complete."
echo "  Log saved to: $LOGFILE"
echo "================================================================================"
