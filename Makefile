# i8gemm Makefile
# Supported targets: test, bench, check, clean
ARCH_FLAGS = -march=armv8.6-a+bf16+i8mm
COMMON_FLAGS = -O2 -Wall
ASM_I8 = i8gemm_k.S
ASM_BF16 = bf16gemm_k.S
ALL_ASM = $(ASM_I8) $(ASM_BF16)

.PHONY: all test bench check clean

all: test bench

test: test_correctness
	./test_correctness 2>&1 | tee test_correctness.log

bench: bench_perf
	./bench_perf shape.csv 2>&1 | tee bench_perf.log

test_correctness: test_correctness.c $(ALL_ASM)
	cc -o $@ $^ $(ARCH_FLAGS) $(COMMON_FLAGS) -lm

bench_perf: bench_perf.c $(ALL_ASM)
	cc -o $@ $^ $(ARCH_FLAGS) $(COMMON_FLAGS) -lm

# Legacy: original i8 test (no tail, no bf16)
calculatei8mm: calculatei8mm.c i8gemm_k_ld.S
	cc -o $@ $^ -march=armv8.6-a+i8mm -O2 -Wall

check: test_correctness
	./test_correctness 2>&1 | tee test_correctness.log

clean:
	rm -f test_correctness bench_perf calculatei8mm *.o *.log
