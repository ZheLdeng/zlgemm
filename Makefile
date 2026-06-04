# i8gemm Makefile
# Supported targets: test, bench, check, clean
#   bench: single-thread shape benchmark + multi-threaded test
ARCH_FLAGS = -march=armv8.6-a+bf16+i8mm
COMMON_FLAGS = -O2 -Wall
ASM_I8 = i8gemm_k.S i8gemm_k_bias.S
ASM_BF16 = bf16gemm_k.S bf16gemm_k_bias.S
ALL_ASM = $(ASM_I8) $(ASM_BF16)
MT_SRC = bf16gemm_mt.c i8gemm_mt.c

.PHONY: all test bench check clean

all: test bench

test: test_correctness
	./test_correctness 2>&1 | tee test_correctness.log

bench: bench_perf
	./bench_perf shape.csv 2>&1 | tee bench_perf.log

# Multi-threaded BF16 GEMM benchmark (via bench_perf --mt)
mt: bench_perf
	./bench_perf --mt 2048 4096 2048

test_correctness: test_correctness.c $(ALL_ASM)
	cc -o $@ $^ $(ARCH_FLAGS) $(COMMON_FLAGS) -lm

# bench_perf links bf16gemm_mt.c for multi-threaded mode
bench_perf: bench_perf.c $(MT_SRC) $(ALL_ASM)
	cc -o $@ $^ $(ARCH_FLAGS) $(COMMON_FLAGS) -fopenmp -lm

# Legacy: original i8 test (no tail, no bf16)
calculatei8mm: calculatei8mm.c i8gemm_k_ld.S
	cc -o $@ $^ -march=armv8.6-a+i8mm -O2 -Wall

check: test_correctness
	./test_correctness 2>&1 | tee test_correctness.log

clean:
	rm -f test_correctness bench_perf calculatei8mm *.o *.log
