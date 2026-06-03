#!/bin/bash
#===============================================================================
# bf16gemm — Correctness & Performance Benchmark
#===============================================================================
set -e

ARCH="-march=armv8.6-a+bf16"
CFLAGS="-O2 -Wall"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
LOGFILE="bench_bf16.log"

exec > >(tee "$LOGFILE") 2>&1

echo "================================================================================"
echo "  BF16 GEMM Kernel Benchmark"
echo "  Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  CPU:  $(lscpu 2>/dev/null | grep 'Model name' | sed 's/Model name:\s*//')"
echo "  Arch: $(uname -m)"
echo "================================================================================"

# ==============================================================================
# 1. Build correctness test
# ==============================================================================
echo ""
echo "── [1/3] Building correctness test ──"
cc -o calculate_bf16_test calculate_bf16mm_ld.c bf16gemm_k_ld.S $ARCH $CFLAGS -lm
echo "  -> calculate_bf16_test built OK"

# ==============================================================================
# 2. Correctness sweep
# ==============================================================================
echo ""
echo "── [2/3] Correctness sweep ──"
RESULT=$(./calculate_bf16_test sweep 2>&1 | tail -1)
echo "  $RESULT"
if echo "$RESULT" | grep -q "0/"; then
    echo "  ERROR: Correctness failures!"
    exit 1
fi

# ==============================================================================
# 3. Build benchmark
# ==============================================================================
echo ""
echo "── [3/3] Building benchmark ──"
cc -o bench_bf16 bench_bf16.c bf16gemm_k_ld.S $ARCH $CFLAGS -lm
echo "  -> bench_bf16 built OK"

run_bench() {
    local m=$1 k=$2 n=$3 loops=$4
    ./bench_bf16 $m $k $n $loops 2>/dev/null
}

divider() {
    printf "  %5s %5s %5s  %10s  %8s\n" "-----" "-----" "-----" "----------" "--------"
}

header() {
    printf "  %5s %5s %5s  %10s  %8s\n" "M" "K" "N" "t(us)" "GFLOPS"
    divider
}

echo ""
echo "── Performance Benchmarks ──"
echo ""

# ---- 3a. Varying N ----
echo "  --- Varying N (M=64 K=128) ---"
header
for N in 8 16 32 64 128 256; do
    run_bench 64 128 $N 200
done
echo ""

# ---- 3b. Varying K ----
echo "  --- Varying K (M=64 N=64) ---"
header
for K in 8 16 32 64 128 256 512; do
    run_bench 64 $K 64 200
done
echo ""

# ---- 3c. Varying M ----
echo "  --- Varying M (K=128 N=64) ---"
header
for M in 8 16 32 64 128 256; do
    run_bench $M 128 64 200
done
echo ""

# ---- 3d. Full-size sweep ----
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

# ---- 3e. Stabilized peak ----
echo "  --- Stabilized peak (3 runs, M=128 K=256 N=128) ---"
header
for i in 1 2 3; do
    run_bench 128 256 128 200
done
divider
echo ""

# ---- 3f. LLM-like shapes ----
echo "  --- LLM-like shapes ---"
header
# Decode: small M, large K & N
for M in 8 16 32; do
    run_bench $M 8192 8192 5
done
echo "  # Prefill:"
for M in 256 512; do
    run_bench $M 4096 4096 3
done

echo ""
echo "================================================================================"
echo "  Benchmark complete. Log saved to: $LOGFILE"
echo "================================================================================"
