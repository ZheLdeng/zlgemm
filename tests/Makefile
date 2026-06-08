SHELL := /bin/bash

CC ?= cc
BUILD_DIR ?= build
LIB_DIR ?= ../lib

ARCH_FLAGS ?= -mcpu=native
OPT_FLAGS ?= -O3
WARN_FLAGS ?= -Wall
OMP_FLAGS ?= -fopenmp
CFLAGS ?= $(OPT_FLAGS) $(WARN_FLAGS) $(OMP_FLAGS) $(ARCH_FLAGS) -I$(LIB_DIR)
LDLIBS ?= -lm

SVE_KERNELS := \
	$(LIB_DIR)/bf16gemm_sve.c \
	$(LIB_DIR)/bf16gemm_sve.S \
	$(LIB_DIR)/i8gemm_sve.c \
	$(LIB_DIR)/i8gemm_sve.S \
	$(LIB_DIR)/i8gemm_pack_a_neon.S
NEON_KERNELS := \
	$(LIB_DIR)/bf16gemm_mt.c \
	$(LIB_DIR)/bf16gemm_k.S \
	$(LIB_DIR)/bf16gemm_k_bias.S \
	$(LIB_DIR)/i8gemm_mt.c \
	$(LIB_DIR)/i8gemm_k.S \
	$(LIB_DIR)/i8gemm_k_bias.S \
	$(LIB_DIR)/i8gemm_pack_a_neon.S

.PHONY: all test test-sve test-neon bench-sve bench-neon bench-nld-sve bench-tp m8-interfaces clean

all: $(BUILD_DIR)/test_correctness_sve $(BUILD_DIR)/bench_perf_sve $(BUILD_DIR)/test_correctness $(BUILD_DIR)/bench_perf

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/test_correctness_sve: test_correctness_sve.c $(SVE_KERNELS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ test_correctness_sve.c $(SVE_KERNELS) $(LDLIBS)

$(BUILD_DIR)/bench_perf_sve: bench_perf_sve.c $(SVE_KERNELS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ bench_perf_sve.c $(SVE_KERNELS) $(LDLIBS)

$(BUILD_DIR)/test_correctness: test_correctness.c $(NEON_KERNELS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ test_correctness.c $(NEON_KERNELS) $(LDLIBS)

$(BUILD_DIR)/bench_perf: bench_perf.c $(NEON_KERNELS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ bench_perf.c $(NEON_KERNELS) $(LDLIBS)

$(BUILD_DIR)/bench_nld_sve: bench_nld_sve.c bf16gemm_sve_nld.S | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ bench_nld_sve.c bf16gemm_sve_nld.S $(LDLIBS)

$(BUILD_DIR)/bench_thread_pool: experimental/bench_thread_pool.c experimental/gemm_thread_pool.c $(SVE_KERNELS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Iexperimental -o $@ experimental/bench_thread_pool.c experimental/gemm_thread_pool.c $(SVE_KERNELS) $(LDLIBS)

$(BUILD_DIR)/test_m8_sve_interfaces: test_sve_m8_interfaces.c bf16gemm_sve_m8_nld.S | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ test_sve_m8_interfaces.c bf16gemm_sve_m8_nld.S $(LDLIBS)

$(BUILD_DIR)/test_m8_neon_interfaces: test_sve_m8_interfaces.c bf16gemm_neon_m8_nld.S | $(BUILD_DIR)
	$(CC) $(CFLAGS) -DM8_NEON_PACK=1 -o $@ test_sve_m8_interfaces.c bf16gemm_neon_m8_nld.S $(LDLIBS)

test: test-sve test-neon

test-sve: $(BUILD_DIR)/test_correctness_sve
	$<

test-neon: $(BUILD_DIR)/test_correctness
	$<

bench-sve: $(BUILD_DIR)/bench_perf_sve
	$< shape.csv

bench-neon: $(BUILD_DIR)/bench_perf
	$< shape.csv

bench-nld-sve: $(BUILD_DIR)/bench_nld_sve
	$<

bench-tp: $(BUILD_DIR)/bench_thread_pool
	$<

m8-interfaces: $(BUILD_DIR)/test_m8_sve_interfaces $(BUILD_DIR)/test_m8_neon_interfaces
	$(BUILD_DIR)/test_m8_sve_interfaces
	$(BUILD_DIR)/test_m8_neon_interfaces

clean:
	rm -rf $(BUILD_DIR)
