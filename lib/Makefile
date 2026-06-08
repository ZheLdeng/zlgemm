SHELL := /bin/bash

CC ?= cc
AR ?= ar
BUILD_DIR ?= build

ARCH_FLAGS ?= -mcpu=native
OPT_FLAGS ?= -O3
WARN_FLAGS ?= -Wall
OMP_FLAGS ?= -fopenmp
PIC_FLAGS ?= -fPIC
CFLAGS ?= $(OPT_FLAGS) $(WARN_FLAGS) $(OMP_FLAGS) $(ARCH_FLAGS) $(PIC_FLAGS) -I.
LDFLAGS ?=
LDLIBS ?= -lm

SVE_STATIC := $(BUILD_DIR)/libi8gemm_sve.a
SVE_SHARED := $(BUILD_DIR)/libi8gemm_sve.so
NEON_STATIC := $(BUILD_DIR)/libi8gemm_neon.a
NEON_SHARED := $(BUILD_DIR)/libi8gemm_neon.so

SVE_OBJS := \
	$(BUILD_DIR)/bf16gemm_sve_c.o \
	$(BUILD_DIR)/bf16gemm_sve_asm.o \
	$(BUILD_DIR)/i8gemm_sve_c.o \
	$(BUILD_DIR)/i8gemm_sve_asm.o \
	$(BUILD_DIR)/i8gemm_pack_a_neon.o

NEON_OBJS := \
	$(BUILD_DIR)/bf16gemm_mt.o \
	$(BUILD_DIR)/i8gemm_mt.o \
	$(BUILD_DIR)/bf16gemm_k.o \
	$(BUILD_DIR)/bf16gemm_k_bias.o \
	$(BUILD_DIR)/i8gemm_k.o \
	$(BUILD_DIR)/i8gemm_k_bias.o \
	$(BUILD_DIR)/i8gemm_pack_a_neon.o

.PHONY: all static shared sve neon clean

all: static shared

static: $(SVE_STATIC)

shared: $(SVE_SHARED)

sve: $(SVE_STATIC) $(SVE_SHARED)

neon: $(NEON_STATIC) $(NEON_SHARED)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/bf16gemm_sve_c.o: bf16gemm_sve.c bf16gemm.h bf16gemm_sve.h gemm_params.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/bf16gemm_sve_asm.o: bf16gemm_sve.S gemm_params.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/i8gemm_sve_c.o: i8gemm_sve.c i8gemm.h gemm_params.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/i8gemm_sve_asm.o: i8gemm_sve.S gemm_params.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/bf16gemm_mt.o: bf16gemm_mt.c bf16gemm.h gemm_params.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/i8gemm_mt.o: i8gemm_mt.c i8gemm.h gemm_params.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/bf16gemm_k.o: bf16gemm_k.S gemm_params.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/bf16gemm_k_bias.o: bf16gemm_k_bias.S gemm_params.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/i8gemm_k.o: i8gemm_k.S gemm_params.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/i8gemm_k_bias.o: i8gemm_k_bias.S gemm_params.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/i8gemm_pack_a_neon.o: i8gemm_pack_a_neon.S | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(SVE_STATIC): $(SVE_OBJS)
	$(AR) rcs $@ $^

$(SVE_SHARED): $(SVE_OBJS)
	$(CC) $(CFLAGS) -shared -o $@ $^ $(LDFLAGS) $(LDLIBS)

$(NEON_STATIC): $(NEON_OBJS)
	$(AR) rcs $@ $^

$(NEON_SHARED): $(NEON_OBJS)
	$(CC) $(CFLAGS) -shared -o $@ $^ $(LDFLAGS) $(LDLIBS)

clean:
	rm -rf $(BUILD_DIR)
