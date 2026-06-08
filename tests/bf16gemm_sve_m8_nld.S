// bf16gemm_sve_m8_nld.S -- M=8 SVE BF16 kernel with reordered A.
//
// ABI matches the other low-level BF16 kernels:
//   x0 A, x1 packed B, x2 C, x3 reordered A, x4 params
//   x5 bias for the bias entry points
//
// This kernel assumes:
//   - M is a multiple of 8
//   - A_reorder is non-NULL and packed as 4 row-pairs per K4 block
//   - C is row-major with params.ldc stride
//   - no C load

	.text
	.align 4

#define A_ADDR_ORG      x0
#define B_ADDR          x1
#define C_ADDR_0        x2
#define A_REORDER_ORG   x3
#define PARAMS          x4

#define M_SIZE          w6
#define A_M_STRIDE      x8
#define K_SIZE          w20
#define K_SIZE_X        x20
#define N_SIZE          w21
#define LDA             w7
#define LDA_B           x7
#define LDC             w23
#define LDC_X           x23

#define N_COUNT         w10
#define B_N_BASE        x11
#define C_N_BASE        x12
#define B_N_STRIDE      x13
#define C_N_STRIDE      x14
#define K_COUNT         w15
#define BIAS_ARG        x5
#define C_PAIR_BASE0    x15
#define C_PAIR_BASE1    x16
#define C_PAIR_BASE2    x17
#define C_PAIR_BASE3    x18
#define C_COL_BASE0     x19
#define C_COL_BASE1     x22
#define C_COL_BASE2     x24
#define C_COL_BASE3     x26
#define C_COL_BASE4     x29
#define C_COL_BASE5     x30
#define BIAS_N_BASE     x23
#define B_K_ADDR        x17
#define A_K_ADDR        x18
#define LDC_B           x25
#define LDC_B_W         w25
#define VL_B            x27
#define A_REORDER_BASE  x28
#define A_REORDER_PTR   x29
#define NTILE           x9
#define NTILE_W         w9

	.macro SAVE_REGS
	stp x19, x20, [sp, #-16]!
	stp x21, x22, [sp, #-16]!
	stp x23, x24, [sp, #-16]!
	stp x25, x26, [sp, #-16]!
	stp x27, x28, [sp, #-16]!
	stp x29, x30, [sp, #-16]!
	stp d8,  d9,  [sp, #-16]!
	stp d10, d11, [sp, #-16]!
	stp d12, d13, [sp, #-16]!
	stp d14, d15, [sp, #-16]!
	.endm

	.macro RESTORE_REGS
	ldp d14, d15, [sp], #16
	ldp d12, d13, [sp], #16
	ldp d10, d11, [sp], #16
	ldp d8,  d9,  [sp], #16
	ldp x29, x30, [sp], #16
	ldp x27, x28, [sp], #16
	ldp x25, x26, [sp], #16
	ldp x23, x24, [sp], #16
	ldp x21, x22, [sp], #16
	ldp x19, x20, [sp], #16
	.endm

	.macro ZERO_ACC
	mov z16.s, #0
	mov z17.s, #0
	mov z18.s, #0
	mov z19.s, #0
	mov z20.s, #0
	mov z21.s, #0
	mov z22.s, #0
	mov z23.s, #0
	mov z24.s, #0
	mov z25.s, #0
	mov z26.s, #0
	mov z27.s, #0
	mov z28.s, #0
	mov z29.s, #0
	mov z30.s, #0
	mov z31.s, #0
	.endm

	.macro INIT_ACC_ZERO
	ZERO_ACC
	.endm

	.macro INIT_ACC_BIAS
	index z9.s, #0, #1
	mov z10.d, z9.d
	and z10.s, z10.s, #3
	lsr z11.s, z9.s, #2
	lsl z11.s, z11.s, #5
	mov z12.d, z10.d
	and z12.s, z12.s, #1
	lsl z12.s, z12.s, #2
	add z15.s, z11.s, z12.s
	mov z13.d, z15.d
	ld1w {z16.s}, p1/z, [BIAS_N_BASE, z13.s, sxtw]
	mov z13.d, z15.d
	add z13.s, z13.s, #8
	ld1w {z17.s}, p1/z, [BIAS_N_BASE, z13.s, sxtw]
	mov z13.d, z15.d
	add z13.s, z13.s, #16
	ld1w {z18.s}, p1/z, [BIAS_N_BASE, z13.s, sxtw]
	mov z13.d, z15.d
	add z13.s, z13.s, #24
	ld1w {z19.s}, p1/z, [BIAS_N_BASE, z13.s, sxtw]
	mov z20.d, z16.d
	mov z21.d, z17.d
	mov z22.d, z18.d
	mov z23.d, z19.d
	mov z24.d, z16.d
	mov z25.d, z17.d
	mov z26.d, z18.d
	mov z27.d, z19.d
	mov z28.d, z16.d
	mov z29.d, z17.d
	mov z30.d, z18.d
	mov z31.d, z19.d
	.endm

	.macro LOAD_AB_PACKED a0, a1, a2, a3, b0, b1, b2, b3
	ld1h {\b0\().h}, p0/z, [B_K_ADDR]
	ld1h {\b1\().h}, p0/z, [B_K_ADDR, #1, mul vl]
	ld1h {\b2\().h}, p0/z, [B_K_ADDR, #2, mul vl]
	ld1h {\b3\().h}, p0/z, [B_K_ADDR, #3, mul vl]
	add B_K_ADDR, B_K_ADDR, VL_B, lsl #2
	ld1rqh {\a0\().h}, p0/z, [A_K_ADDR]
	ld1rqh {\a1\().h}, p0/z, [A_K_ADDR, #16]
	ld1rqh {\a2\().h}, p0/z, [A_K_ADDR, #32]
	ld1rqh {\a3\().h}, p0/z, [A_K_ADDR, #48]
	add A_K_ADDR, A_K_ADDR, #64
	.endm

	.macro REORDER_A_M8
	mov A_REORDER_PTR, A_REORDER_BASE
	mov A_K_ADDR, A_ADDR_ORG
	mov K_COUNT, K_SIZE
1:
	add x11, A_K_ADDR, LDA_B
	add x12, A_K_ADDR, LDA_B, lsl #1
	add x16, A_K_ADDR, LDA_B, lsl #2
	sub x17, x16, LDA_B
	add x19, x11, LDA_B, lsl #2
	add x22, x12, LDA_B, lsl #2
	add x24, x19, LDA_B, lsl #1
	ldr d0, [A_K_ADDR]
	ld1 {v0.d}[1], [x11]
	ldr d1, [x12]
	ld1 {v1.d}[1], [x17]
	ldr d2, [x16]
	ld1 {v2.d}[1], [x19]
	ldr d3, [x22]
	ld1 {v3.d}[1], [x24]
	stp q0, q1, [A_REORDER_PTR], #32
	stp q2, q3, [A_REORDER_PTR], #32
	add A_K_ADDR, A_K_ADDR, #8
	subs K_COUNT, K_COUNT, #4
	b.gt 1b
	.endm

	.macro BFMMLA_BLOCK_AB a0, a1, a2, a3, b0, b1, b2, b3
	bfmmla z16.s, \a0\().h, \b0\().h
	bfmmla z20.s, \a1\().h, \b0\().h
	bfmmla z24.s, \a2\().h, \b0\().h
	bfmmla z28.s, \a3\().h, \b0\().h
	bfmmla z17.s, \a0\().h, \b1\().h
	bfmmla z21.s, \a1\().h, \b1\().h
	bfmmla z25.s, \a2\().h, \b1\().h
	bfmmla z29.s, \a3\().h, \b1\().h
	bfmmla z18.s, \a0\().h, \b2\().h
	bfmmla z22.s, \a1\().h, \b2\().h
	bfmmla z26.s, \a2\().h, \b2\().h
	bfmmla z30.s, \a3\().h, \b2\().h
	bfmmla z19.s, \a0\().h, \b3\().h
	bfmmla z23.s, \a1\().h, \b3\().h
	bfmmla z27.s, \a2\().h, \b3\().h
	bfmmla z31.s, \a3\().h, \b3\().h
	.endm

	.macro LOAD_A0_B0
	LOAD_AB_PACKED z0, z1, z2, z3, z4, z5, z6, z7
	.endm

	.macro LOAD_A1_B1_COMPUTE_A0_B0
	ld1h {z12.h}, p0/z, [B_K_ADDR]
	ld1h {z13.h}, p0/z, [B_K_ADDR, #1, mul vl]
	ld1h {z14.h}, p0/z, [B_K_ADDR, #2, mul vl]
	ld1h {z15.h}, p0/z, [B_K_ADDR, #3, mul vl]
	bfmmla z16.s, z0.h, z4.h
	bfmmla z20.s, z1.h, z4.h
	bfmmla z24.s, z2.h, z4.h
	bfmmla z28.s, z3.h, z4.h
	bfmmla z17.s, z0.h, z5.h
	bfmmla z21.s, z1.h, z5.h
	bfmmla z25.s, z2.h, z5.h
	bfmmla z29.s, z3.h, z5.h
	add B_K_ADDR, B_K_ADDR, VL_B, lsl #2
	ld1rqh {z8.h}, p0/z, [A_K_ADDR]
	ld1rqh {z9.h}, p0/z, [A_K_ADDR, #16]
	ld1rqh {z10.h}, p0/z, [A_K_ADDR, #32]
	ld1rqh {z11.h}, p0/z, [A_K_ADDR, #48]
	bfmmla z18.s, z0.h, z6.h
	bfmmla z22.s, z1.h, z6.h
	bfmmla z26.s, z2.h, z6.h
	bfmmla z30.s, z3.h, z6.h
	bfmmla z19.s, z0.h, z7.h
	bfmmla z23.s, z1.h, z7.h
	add A_K_ADDR, A_K_ADDR, #64
	bfmmla z27.s, z2.h, z7.h
	bfmmla z31.s, z3.h, z7.h
	.endm

	.macro LOAD_A0_B0_COMPUTE_A1_B1
	ld1h {z4.h}, p0/z, [B_K_ADDR]
	ld1h {z5.h}, p0/z, [B_K_ADDR, #1, mul vl]
	ld1h {z6.h}, p0/z, [B_K_ADDR, #2, mul vl]
	ld1h {z7.h}, p0/z, [B_K_ADDR, #3, mul vl]
	bfmmla z16.s, z8.h, z12.h
	bfmmla z20.s, z9.h, z12.h
	bfmmla z24.s, z10.h, z12.h
	bfmmla z28.s, z11.h, z12.h
	bfmmla z17.s, z8.h, z13.h
	bfmmla z21.s, z9.h, z13.h
	bfmmla z25.s, z10.h, z13.h
	bfmmla z29.s, z11.h, z13.h
	add B_K_ADDR, B_K_ADDR, VL_B, lsl #2
	ld1rqh {z0.h}, p0/z, [A_K_ADDR]
	ld1rqh {z1.h}, p0/z, [A_K_ADDR, #16]
	ld1rqh {z2.h}, p0/z, [A_K_ADDR, #32]
	ld1rqh {z3.h}, p0/z, [A_K_ADDR, #48]
	bfmmla z18.s, z8.h, z14.h
	bfmmla z22.s, z9.h, z14.h
	bfmmla z26.s, z10.h, z14.h
	bfmmla z30.s, z11.h, z14.h
	bfmmla z19.s, z8.h, z15.h
	bfmmla z23.s, z9.h, z15.h
	add A_K_ADDR, A_K_ADDR, #64
	bfmmla z27.s, z10.h, z15.h
	bfmmla z31.s, z11.h, z15.h
	.endm

	.macro COMPUTE_A0_B0
	BFMMLA_BLOCK_AB z0, z1, z2, z3, z4, z5, z6, z7
	.endm

	.macro COMPUTE_A1_B1
	BFMMLA_BLOCK_AB z8, z9, z10, z11, z12, z13, z14, z15
	.endm

	.macro BUILD_F32_OFFSETS_AND_BASES
	index z9.s, #0, #1
	mov C_PAIR_BASE0, C_N_BASE
	mov z10.d, z9.d
	and z10.s, z10.s, #3
	add C_PAIR_BASE1, C_N_BASE, LDC_B, lsl #1
	lsr z11.s, z9.s, #2
	add C_PAIR_BASE2, C_N_BASE, LDC_B, lsl #2
	mov z12.d, z10.d
	and z12.s, z12.s, #1
	add C_PAIR_BASE3, C_PAIR_BASE2, LDC_B, lsl #1
	lsr z13.s, z10.s, #1
	lsl z11.s, z11.s, #5
	lsl z12.s, z12.s, #2
	and z13.s, z13.s, #1
	dup z14.s, LDC_B_W
	mul z13.s, p1/m, z13.s, z14.s
	add z9.s, z11.s, z12.s
	mov z15.d, z9.d
	add z9.s, z9.s, z13.s
	.endm

	.macro BUILD_BF16_OFFSETS_AND_BASES
	index z9.s, #0, #1
	lsr C_COL_BASE3, LDC_B, #1
	mov C_PAIR_BASE0, C_N_BASE
	mov z10.d, z9.d
	and z10.s, z10.s, #3
	add C_PAIR_BASE1, C_N_BASE, C_COL_BASE3, lsl #1
	lsr z11.s, z9.s, #2
	add C_PAIR_BASE2, C_N_BASE, C_COL_BASE3, lsl #2
	mov z12.d, z10.d
	and z12.s, z12.s, #1
	add C_PAIR_BASE3, C_PAIR_BASE2, C_COL_BASE3, lsl #1
	lsr z13.s, z10.s, #1
	lsl z14.s, z11.s, #5
	lsl z15.s, z12.s, #2
	add z15.s, z14.s, z15.s
	lsl z11.s, z11.s, #4
	lsl z12.s, z12.s, #1
	and z13.s, z13.s, #1
	dup z14.s, w26
	mul z13.s, p1/m, z13.s, z14.s
	add z9.s, z11.s, z12.s
	add z9.s, z9.s, z13.s
	.endm

	.macro STORE_8XNTILE_F32
	mov C_COL_BASE0, C_PAIR_BASE0
	add C_COL_BASE1, C_PAIR_BASE0, #8
	add C_COL_BASE2, C_PAIR_BASE0, #16
	add C_COL_BASE3, C_PAIR_BASE0, #24
	mov C_COL_BASE4, C_PAIR_BASE1
	add C_COL_BASE5, C_PAIR_BASE1, #8
	st1w {z16.s}, p1, [C_COL_BASE0, z9.s, sxtw]
	st1w {z17.s}, p1, [C_COL_BASE1, z9.s, sxtw]
	add C_COL_BASE0, C_PAIR_BASE1, #16
	add C_COL_BASE1, C_PAIR_BASE1, #24
	st1w {z18.s}, p1, [C_COL_BASE2, z9.s, sxtw]
	st1w {z19.s}, p1, [C_COL_BASE3, z9.s, sxtw]
	mov C_COL_BASE2, C_PAIR_BASE2
	add C_COL_BASE3, C_PAIR_BASE2, #8
	st1w {z20.s}, p1, [C_COL_BASE4, z9.s, sxtw]
	st1w {z21.s}, p1, [C_COL_BASE5, z9.s, sxtw]
	add C_COL_BASE4, C_PAIR_BASE2, #16
	add C_COL_BASE5, C_PAIR_BASE2, #24
	st1w {z22.s}, p1, [C_COL_BASE0, z9.s, sxtw]
	st1w {z23.s}, p1, [C_COL_BASE1, z9.s, sxtw]
	mov C_COL_BASE0, C_PAIR_BASE3
	add C_COL_BASE1, C_PAIR_BASE3, #8
	st1w {z24.s}, p1, [C_COL_BASE2, z9.s, sxtw]
	st1w {z25.s}, p1, [C_COL_BASE3, z9.s, sxtw]
	add C_COL_BASE2, C_PAIR_BASE3, #16
	add C_COL_BASE3, C_PAIR_BASE3, #24
	st1w {z26.s}, p1, [C_COL_BASE4, z9.s, sxtw]
	st1w {z27.s}, p1, [C_COL_BASE5, z9.s, sxtw]
	st1w {z28.s}, p1, [C_COL_BASE0, z9.s, sxtw]
	st1w {z29.s}, p1, [C_COL_BASE1, z9.s, sxtw]
	st1w {z30.s}, p1, [C_COL_BASE2, z9.s, sxtw]
	st1w {z31.s}, p1, [C_COL_BASE3, z9.s, sxtw]
	.endm

	.macro STORE_BF16_AT base, c_imm, acc
	add C_COL_BASE0, \base, \c_imm
	bfcvt z14.h, p1/m, \acc\().s
	st1h {z14.s}, p1, [C_COL_BASE0, z9.s, sxtw]
	.endm

	.macro STORE_8XNTILE_BF16
	STORE_BF16_AT C_PAIR_BASE0, #0,  z16
	STORE_BF16_AT C_PAIR_BASE0, #4,  z17
	STORE_BF16_AT C_PAIR_BASE0, #8,  z18
	STORE_BF16_AT C_PAIR_BASE0, #12, z19
	STORE_BF16_AT C_PAIR_BASE1, #0,  z20
	STORE_BF16_AT C_PAIR_BASE1, #4,  z21
	STORE_BF16_AT C_PAIR_BASE1, #8,  z22
	STORE_BF16_AT C_PAIR_BASE1, #12, z23
	STORE_BF16_AT C_PAIR_BASE2, #0,  z24
	STORE_BF16_AT C_PAIR_BASE2, #4,  z25
	STORE_BF16_AT C_PAIR_BASE2, #8,  z26
	STORE_BF16_AT C_PAIR_BASE2, #12, z27
	STORE_BF16_AT C_PAIR_BASE3, #0,  z28
	STORE_BF16_AT C_PAIR_BASE3, #4,  z29
	STORE_BF16_AT C_PAIR_BASE3, #8,  z30
	STORE_BF16_AT C_PAIR_BASE3, #12, z31
	.endm

	.macro STORE_F32_M8
	BUILD_F32_OFFSETS_AND_BASES
	STORE_8XNTILE_F32
	.endm

	.macro STORE_BF16_M8
	BUILD_BF16_OFFSETS_AND_BASES
	STORE_8XNTILE_BF16
	.endm

	.macro BIAS_RESET_NONE
	.endm

	.macro BIAS_RESET_F32
	mov BIAS_N_BASE, BIAS_ARG
	.endm

	.macro BIAS_ADVANCE_NONE
	.endm

	.macro BIAS_ADVANCE_F32
	add BIAS_N_BASE, BIAS_N_BASE, C_N_STRIDE
	.endm

	.macro ADVANCE_C_N_F32
	add C_N_BASE, C_N_BASE, C_N_STRIDE
	.endm

	.macro ADVANCE_C_N_BF16
	add C_N_BASE, C_N_BASE, VL_B
	.endm

	.macro ADVANCE_C_M_F32
	add C_ADDR_0, C_ADDR_0, LDC_B, lsl #3
	.endm

	.macro ADVANCE_C_M_BF16
	add C_ADDR_0, C_ADDR_0, LDC_B, lsl #2
	.endm

	.macro GEMM_BODY_M8 func_name, init_acc, store_c, bias_reset, bias_advance, advance_c_n, advance_c_m
	.global \func_name
	.type \func_name, %function
	.p2align 4
\func_name:
	SAVE_REGS

	ldr M_SIZE, [PARAMS]
	ldr K_SIZE, [PARAMS, #4]
	ldr N_SIZE, [PARAMS, #8]
	ldr LDA, [PARAMS, #12]
	ldr LDC, [PARAMS, #20]
	mov A_REORDER_BASE, A_REORDER_ORG

	cntb VL_B
	lsr NTILE, VL_B, #1
	lsl LDA_B, LDA_B, #1
	mul B_N_STRIDE, K_SIZE_X, VL_B
	lsl LDC_B, LDC_X, #2
	lsl C_N_STRIDE, VL_B, #1
	lsl A_M_STRIDE, K_SIZE_X, #4
	ptrue p0.b
	ptrue p1.s

.L_m_loop_\@:
	cmp M_SIZE, #0
	b.le .L_done_\@
	REORDER_A_M8
	mov B_N_BASE, B_ADDR
	mov C_N_BASE, C_ADDR_0
	\bias_reset
	mov N_COUNT, N_SIZE

.L_n_loop_\@:
	cmp N_COUNT, #0
	b.le .L_m_next_\@

	\init_acc
	mov B_K_ADDR, B_N_BASE
	mov A_K_ADDR, A_REORDER_BASE
	mov K_COUNT, K_SIZE

		LOAD_A0_B0
	subs K_COUNT, K_COUNT, #4
	b.eq .L_cached_tail_cur_\@

.L_k_loop_\@:
		LOAD_A1_B1_COMPUTE_A0_B0
	subs K_COUNT, K_COUNT, #4
	b.eq .L_cached_tail_next_\@
		LOAD_A0_B0_COMPUTE_A1_B1
	subs K_COUNT, K_COUNT, #4
	b.gt .L_k_loop_\@

.L_cached_tail_cur_\@:
		COMPUTE_A0_B0
	b .L_k_done_\@

.L_cached_tail_next_\@:
		COMPUTE_A1_B1

.L_k_done_\@:
	\store_c
	add B_N_BASE, B_N_BASE, B_N_STRIDE
	\bias_advance
	\advance_c_n
	sub N_COUNT, N_COUNT, NTILE_W
	b .L_n_loop_\@

.L_m_next_\@:
	add A_REORDER_BASE, A_REORDER_BASE, A_M_STRIDE
	add A_ADDR_ORG, A_ADDR_ORG, LDA_B, lsl #3
	\advance_c_m
	subs M_SIZE, M_SIZE, #8
	b.gt .L_m_loop_\@

.L_done_\@:
	RESTORE_REGS
	ret
	.size \func_name, .-\func_name
	.endm

	GEMM_BODY_M8 bf16gemm_k_nld_f_m8, INIT_ACC_ZERO, STORE_F32_M8, BIAS_RESET_NONE, BIAS_ADVANCE_NONE, ADVANCE_C_N_F32, ADVANCE_C_M_F32
	GEMM_BODY_M8 bf16gemm_k_nld_b_m8, INIT_ACC_ZERO, STORE_BF16_M8, BIAS_RESET_NONE, BIAS_ADVANCE_NONE, ADVANCE_C_N_BF16, ADVANCE_C_M_BF16
	GEMM_BODY_M8 bf16gemm_k_nld_bias_f_m8, INIT_ACC_BIAS, STORE_F32_M8, BIAS_RESET_F32, BIAS_ADVANCE_F32, ADVANCE_C_N_F32, ADVANCE_C_M_F32

	.section .note.GNU-stack,"",@progbits
