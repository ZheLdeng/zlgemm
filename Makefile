# ═══════════════════════════════════════════════════════════════════════════
# i8gemm Makefile
# ═══════════════════════════════════════════════════════════════════════════
#
# Targets:
#   all            — build + run test and bench (default)
#   test           — correctness: bf16 (fp32/bf16/bias_f/nld_b) + i8 (i32/fp32/bias_f)
#   bench          — single-thread per-shape benchmark (reads shape.csv)
#                      CORE=n   bind to core n (default 0)
#   mt             — multi-threaded sweep (bf16+i8, M=K=N=2048, taskset all cores)
#   mt-bf16        — multi-threaded sweep bf16 only
#   mt-i8          — multi-threaded sweep i8 only
#                      CORES=64-78  bind to range
#                      CORES=64,67,68  bind to list
#                      (default: 0-$(nproc-1), all cores)
#   check          — same as test (correctness + log)
#   clean          — remove binaries, objects, logs
#
# Sweep sequence: {1,2,4,8,10,16,20,32,40,64}, skipping nthreads >= ncores.
# Core affinity: taskset + OMP_PLACES=cores OMP_PROC_BIND=close.
#
# Source groups:
#   ASM_I8   = i8gemm_k.S i8gemm_k_bias.S        (int8  GEMM kernels)
#   ASM_BF16 = bf16gemm_k.S bf16gemm_k_bias.S     (bf16  GEMM kernels)
#   MT_SRC   = bf16gemm_mt.c i8gemm_mt.c          (multi-threading dispatchers)
#
# Build requirements: ARMv8.6-A + BF16 + I8MM, OpenMP, math lib.
# ═══════════════════════════════════════════════════════════════════════════
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c

CC ?= cc
ARCH_FLAGS ?= -march=armv8.6-a+bf16+i8mm
COMMON_FLAGS ?= -O2 -Wall
CFLAGS ?= $(ARCH_FLAGS) $(COMMON_FLAGS)
LDFLAGS ?=
LDLIBS ?= -lm
OMP_FLAGS ?= -fopenmp
ASM_I8 = i8gemm_k.S i8gemm_k_bias.S
ASM_BF16 = bf16gemm_k.S bf16gemm_k_bias.S
ALL_ASM = $(ASM_I8) $(ASM_BF16)
MT_SRC = bf16gemm_mt.c i8gemm_mt.c
COMMON_HEADERS = gemm_params.h

# ── CPU binding ──
# c  overrides core(s) for taskset:  make bench c=65   make mt c=64-78   make mt c=64,67,68
# Default: bench → core 0;  mt → all cores (0..nproc-1).
NPROC := $(shell nproc 2>/dev/null || echo 64)
LAST  := $(shell echo $$(( $(NPROC) - 1 )))
c     ?=

.PHONY: all test bench mt mt-bf16 mt-i8 check clean

all: test bench

test: test_correctness
	./test_correctness 2>&1 | tee test_correctness.log

bench: bench_perf
	taskset -c $(or $(c),0) ./bench_perf $(CSV) 2>&1 | tee bench_perf.log

# ── Multi-threaded sweep targets ──
# Sweep all shapes from shape.csv, each with all thread counts < ncores.
# Override CSV with e.g.  make bench CSV=my_shapes.csv  or  make mt CSV=my_shapes.csv
CSV ?= shape.csv

mt: bench_perf
	taskset -c $(or $(c),0-$(LAST)) ./bench_perf --mt-sweep-csv-both $(CSV) 2>&1 | tee mt_sweep.log

mt-bf16: bench_perf
	taskset -c $(or $(c),0-$(LAST)) ./bench_perf --mt-sweep-csv $(CSV) 2>&1 | tee mt_sweep.log

mt-i8: bench_perf
	taskset -c $(or $(c),0-$(LAST)) ./bench_perf --mt-sweep-csv-i8 $(CSV) 2>&1 | tee mt_sweep.log

test_correctness: test_correctness.c $(COMMON_HEADERS) $(ALL_ASM)
	$(CC) -o $@ $(filter %.c %.S,$^) $(CFLAGS) $(LDFLAGS) $(LDLIBS)

# bench_perf links bf16gemm_mt.c for multi-threaded mode
bench_perf: bench_perf.c $(MT_SRC) $(COMMON_HEADERS) i8gemm.h bf16gemm.h $(ALL_ASM)
	$(CC) -o $@ $(filter %.c %.S,$^) $(CFLAGS) $(OMP_FLAGS) $(LDFLAGS) $(LDLIBS)

# Legacy: original i8 test (no tail, no bf16)
calculatei8mm: calculatei8mm.c i8gemm_k_ld.S
	$(CC) -o $@ $^ -march=armv8.6-a+i8mm $(COMMON_FLAGS)

check: test_correctness
	./test_correctness 2>&1 | tee test_correctness.log

clean:
	rm -f test_correctness bench_perf calculatei8mm *.o *.log
