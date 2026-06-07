// bf16gemm_sve.S -- SVE BF16 GEMM kernels.
//
// Low-level ABI matches bf16gemm_k.S:
//   x0 A, x1 packed B, x2 C, x3 A_reorder, x4 params
// Bias variants additionally receive x5 = bias.

	.text
	.align 4

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

	.macro LOAD_C_PAIR_F32 base, a0, a1, a2, a3
	ld1w {\a0\().s}, p1/z, [\base, z9.s, sxtw]
	add x3, \base, #8
	ld1w {\a1\().s}, p1/z, [x3, z9.s, sxtw]
	add x3, \base, #16
	ld1w {\a2\().s}, p1/z, [x3, z9.s, sxtw]
	add x3, \base, #24
	ld1w {\a3\().s}, p1/z, [x3, z9.s, sxtw]
	.endm

	.macro LOAD_C_ALL_F32
	ZERO_ACC
	.endm

	.macro LOAD_A_PAIR_TAIL zd, base, row
	cmp w19, #\row
	b.le 920f
	cmp w19, #(\row + 1)
	b.eq 921f
	ld1d {\zd\().d}, p2/z, [\base, z8.d]
	b 922f
921:
	ld1d {\zd\().d}, p3/z, [\base, z8.d]
	b 922f
920:
	mov \zd\().d, #0
922:
	.endm

	.macro BFMMLA_BLOCK
	bfmmla z16.s, z0.h, z4.h
	bfmmla z20.s, z1.h, z4.h
	bfmmla z24.s, z2.h, z4.h
	bfmmla z28.s, z3.h, z4.h
	bfmmla z17.s, z0.h, z5.h
	bfmmla z21.s, z1.h, z5.h
	bfmmla z25.s, z2.h, z5.h
	bfmmla z29.s, z3.h, z5.h
	bfmmla z18.s, z0.h, z6.h
	bfmmla z22.s, z1.h, z6.h
	bfmmla z26.s, z2.h, z6.h
	bfmmla z30.s, z3.h, z6.h
	bfmmla z19.s, z0.h, z7.h
	bfmmla z23.s, z1.h, z7.h
	bfmmla z27.s, z2.h, z7.h
	bfmmla z31.s, z3.h, z7.h
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

	.macro ZERO_ACC_M4
	mov z16.s, #0
	mov z17.s, #0
	mov z18.s, #0
	mov z19.s, #0
	mov z20.s, #0
	mov z21.s, #0
	mov z22.s, #0
	mov z23.s, #0
	.endm

	.macro ZERO_ACC_M2
	mov z16.s, #0
	mov z17.s, #0
	mov z18.s, #0
	mov z19.s, #0
	.endm

	.macro LOAD_AB_PACKED_M4 a0, a1, b0, b1, b2, b3
	ld1h {\b0\().h}, p0/z, [x17]
	ld1h {\b1\().h}, p0/z, [x17, #1, mul vl]
	ld1h {\b2\().h}, p0/z, [x17, #2, mul vl]
	ld1h {\b3\().h}, p0/z, [x17, #3, mul vl]
	add x17, x17, x27, lsl #2
	ld1rqh {\a0\().h}, p0/z, [x18]
	ld1rqh {\a1\().h}, p0/z, [x18, #16]
	add x18, x18, #64
	.endm

	.macro LOAD_AB_PACKED_M2 a0, b0, b1, b2, b3
	ld1h {\b0\().h}, p0/z, [x17]
	ld1h {\b1\().h}, p0/z, [x17, #1, mul vl]
	ld1h {\b2\().h}, p0/z, [x17, #2, mul vl]
	ld1h {\b3\().h}, p0/z, [x17, #3, mul vl]
	add x17, x17, x27, lsl #2
	ld1rqh {\a0\().h}, p0/z, [x18]
	add x18, x18, #64
	.endm

	.macro BFMMLA_M4_AB a0, a1, b0, b1, b2, b3
	bfmmla z16.s, \a0\().h, \b0\().h
	bfmmla z20.s, \a1\().h, \b0\().h
	bfmmla z17.s, \a0\().h, \b1\().h
	bfmmla z21.s, \a1\().h, \b1\().h
	bfmmla z18.s, \a0\().h, \b2\().h
	bfmmla z22.s, \a1\().h, \b2\().h
	bfmmla z19.s, \a0\().h, \b3\().h
	bfmmla z23.s, \a1\().h, \b3\().h
	.endm

	.macro BFMMLA_M2_AB a0, b0, b1, b2, b3
	bfmmla z16.s, \a0\().h, \b0\().h
	bfmmla z17.s, \a0\().h, \b1\().h
	bfmmla z18.s, \a0\().h, \b2\().h
	bfmmla z19.s, \a0\().h, \b3\().h
	.endm

	.macro LOAD_AB_PACKED a0, a1, a2, a3, b0, b1, b2, b3
	ld1h {\b0\().h}, p0/z, [x17]
	ld1h {\b1\().h}, p0/z, [x17, #1, mul vl]
	ld1h {\b2\().h}, p0/z, [x17, #2, mul vl]
	ld1h {\b3\().h}, p0/z, [x17, #3, mul vl]
	add x17, x17, x27, lsl #2
	ld1rqh {\a0\().h}, p0/z, [x18]
	ld1rqh {\a1\().h}, p0/z, [x18, #16]
	ld1rqh {\a2\().h}, p0/z, [x18, #32]
	ld1rqh {\a3\().h}, p0/z, [x18, #48]
	add x18, x18, #64
	.endm

	.macro BUILD_C_OFFSETS
	index z14.s, #0, #1
	mov z15.d, z14.d
	and z15.s, z15.s, #3
	mov z9.d, z14.d
	lsr z9.s, z9.s, #2
	lsl z9.s, z9.s, #5
	mov z13.d, z15.d
	and z13.s, z13.s, #1
	lsl z13.s, z13.s, #2
	add z9.s, z9.s, z13.s
	mov z10.d, z9.d
	mov z12.d, z15.d
	lsr z12.s, z12.s, #1
	and z12.s, z12.s, #1
	cmpeq p4.s, p1/z, z12.s, #0
	dup z13.s, w25
	mul z12.s, p1/m, z12.s, z13.s
	add z9.s, z9.s, z12.s
	mov z11.d, z14.d
	lsr z11.s, z11.s, #2
	lsl z11.s, z11.s, #4
	mov z13.d, z15.d
	and z13.s, z13.s, #1
	lsl z13.s, z13.s, #1
	add z11.s, z11.s, z13.s
	mov z12.d, z15.d
	lsr z12.s, z12.s, #1
	and z12.s, z12.s, #1
	dup z13.s, w26
	mul z12.s, p1/m, z12.s, z13.s
	add z11.s, z11.s, z12.s
	.endm

	.macro ZERO_ACC_M12
	mov z8.s, #0
	mov z9.s, #0
	mov z10.s, #0
	mov z11.s, #0
	mov z12.s, #0
	mov z13.s, #0
	mov z14.s, #0
	mov z15.s, #0
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

	.macro BFMMLA_M12_ROW a, c0, c1, c2, c3
	bfmmla \c0\().s, \a\().h, z4.h
	bfmmla \c1\().s, \a\().h, z5.h
	bfmmla \c2\().s, \a\().h, z6.h
	bfmmla \c3\().s, \a\().h, z7.h
	.endm

	.macro BFMMLA_M12_2ROWS a0, a1, c00, c01, c02, c03, c10, c11, c12, c13
	bfmmla \c00\().s, \a0\().h, z4.h
	bfmmla \c10\().s, \a1\().h, z4.h
	bfmmla \c01\().s, \a0\().h, z5.h
	bfmmla \c11\().s, \a1\().h, z5.h
	bfmmla \c02\().s, \a0\().h, z6.h
	bfmmla \c12\().s, \a1\().h, z6.h
	bfmmla \c03\().s, \a0\().h, z7.h
	bfmmla \c13\().s, \a1\().h, z7.h
	.endm

	.macro BFMMLA_M12_4A
	bfmmla z8.s,  z0.h, z4.h
	bfmmla z12.s, z1.h, z4.h
	bfmmla z16.s, z2.h, z4.h
	bfmmla z20.s, z3.h, z4.h
	bfmmla z9.s,  z0.h, z5.h
	bfmmla z13.s, z1.h, z5.h
	bfmmla z17.s, z2.h, z5.h
	bfmmla z21.s, z3.h, z5.h
	bfmmla z10.s, z0.h, z6.h
	bfmmla z14.s, z1.h, z6.h
	bfmmla z18.s, z2.h, z6.h
	bfmmla z22.s, z3.h, z6.h
	bfmmla z11.s, z0.h, z7.h
	bfmmla z15.s, z1.h, z7.h
	bfmmla z19.s, z2.h, z7.h
	bfmmla z23.s, z3.h, z7.h
	.endm

	.macro BFMMLA_M12_2A
	bfmmla z24.s, z0.h, z4.h
	bfmmla z28.s, z1.h, z4.h
	bfmmla z25.s, z0.h, z5.h
	bfmmla z29.s, z1.h, z5.h
	bfmmla z26.s, z0.h, z6.h
	bfmmla z30.s, z1.h, z6.h
	bfmmla z27.s, z0.h, z7.h
	bfmmla z31.s, z1.h, z7.h
	.endm

	.macro BUILD_F32_OFFSETS_M12
	index z0.s, #0, #1
	mov z1.d, z0.d
	and z1.s, z1.s, #3
	mov z2.d, z0.d
	lsr z2.s, z2.s, #2
	lsl z2.s, z2.s, #5
	mov z3.d, z1.d
	and z3.s, z3.s, #1
	lsl z3.s, z3.s, #2
	add z0.s, z2.s, z3.s
	mov z2.d, z1.d
	lsr z2.s, z2.s, #1
	and z2.s, z2.s, #1
	dup z3.s, w25
	mul z2.s, p1/m, z2.s, z3.s
	add z0.s, z0.s, z2.s
	.endm

	.macro STORE_PAIR_M12 base, a0, a1, a2, a3
	add x3, \base, #0
	st1w {\a0\().s}, p1, [x3, z0.s, sxtw]
	add x3, \base, #8
	st1w {\a1\().s}, p1, [x3, z0.s, sxtw]
	add x3, \base, #16
	st1w {\a2\().s}, p1, [x3, z0.s, sxtw]
	add x3, \base, #24
	st1w {\a3\().s}, p1, [x3, z0.s, sxtw]
	.endm

	.macro M12_K4_BODY
	ld1h {z4.h}, p0/z, [x17]
	ld1h {z5.h}, p0/z, [x17, #1, mul vl]
	ld1h {z6.h}, p0/z, [x17, #2, mul vl]
	ld1h {z7.h}, p0/z, [x17, #3, mul vl]
	add x17, x17, x27, lsl #2
	ld1rqh {z0.h}, p0/z, [x18]
	ld1rqh {z1.h}, p0/z, [x18, #16]
	ld1rqh {z2.h}, p0/z, [x18, #32]
	ld1rqh {z3.h}, p0/z, [x18, #48]
	BFMMLA_M12_2ROWS z0, z1, z8, z9, z10, z11, z12, z13, z14, z15
	ld1rqh {z0.h}, p0/z, [x18, #64]
	ld1rqh {z1.h}, p0/z, [x18, #80]
	BFMMLA_M12_2ROWS z2, z3, z16, z17, z18, z19, z20, z21, z22, z23
	BFMMLA_M12_2A
	add x18, x18, #96
	.endm

	.macro ADD_BIAS_ACC acc, cpbytes
	cbz x30, 930f
	add x3, x30, #\cpbytes
	ld1w {z12.s}, p1/z, [x3, z10.s, sxtw]
	fadd \acc\().s, \acc\().s, z12.s
930:
	.endm

	.macro STORE_ONE_F32 base, pred, acc, cpbytes
	ADD_BIAS_ACC \acc, \cpbytes
	add x3, \base, #\cpbytes
	st1w {\acc\().s}, \pred, [x3, z9.s, sxtw]
	.endm

	.macro STORE_ONE_BF16 base, pred, acc, cpbytes
	ADD_BIAS_ACC \acc, \cpbytes
	bfcvt z13.h, p1/m, \acc\().s
	.if \cpbytes == 0
	add x3, \base, #0
	.elseif \cpbytes == 8
	add x3, \base, #4
	.elseif \cpbytes == 16
	add x3, \base, #8
	.else
	add x3, \base, #12
	.endif
	st1h {z13.s}, \pred, [x3, z11.s, sxtw]
	.endm

	.macro STORE_PAIR base, row, a0, a1, a2, a3
	cmp w19, #\row
	b.le 940f
	cmp w19, #(\row + 1)
	b.eq 941f
	cmp w7, #0
	b.ne 942f
	STORE_ONE_F32 \base, p1, \a0, 0
	STORE_ONE_F32 \base, p1, \a1, 8
	STORE_ONE_F32 \base, p1, \a2, 16
	STORE_ONE_F32 \base, p1, \a3, 24
	b 940f
942:
	STORE_ONE_BF16 \base, p1, \a0, 0
	STORE_ONE_BF16 \base, p1, \a1, 8
	STORE_ONE_BF16 \base, p1, \a2, 16
	STORE_ONE_BF16 \base, p1, \a3, 24
	b 940f
941:
	cmp w7, #0
	b.ne 943f
	STORE_ONE_F32 \base, p4, \a0, 0
	STORE_ONE_F32 \base, p4, \a1, 8
	STORE_ONE_F32 \base, p4, \a2, 16
	STORE_ONE_F32 \base, p4, \a3, 24
	b 940f
943:
	STORE_ONE_BF16 \base, p4, \a0, 0
	STORE_ONE_BF16 \base, p4, \a1, 8
	STORE_ONE_BF16 \base, p4, \a2, 16
	STORE_ONE_BF16 \base, p4, \a3, 24
940:
	.endm

	.p2align 4
bf16gemm_sve_kernel_asm:
	SAVE_REGS

	ldp w19, w20, [x4]        // M, K
	ldr w21, [x4, #8]         // N
	ldr w22, [x4, #12]        // lda
	ldr w23, [x4, #20]        // ldc
	mov x28, x3               // packed A, or NULL for gather fallback

	cntb x27
	lsr x9, x27, #1           // N tile in columns: VL bytes / 2
	lsl x24, x22, #1          // lda bytes
	lsl x25, x23, #2          // f32 ldc bytes
	lsl x26, x23, #1          // bf16 ldc bytes
	mul x13, x20, x27         // packed-B N-tile stride: K * VL bytes
	lsl x14, x27, #1          // f32 C N-tile advance

	ptrue p0.b
	ptrue p1.s
	ptrue p2.d

	// A gather offsets: dword lanes alternate [row0, row1].
	index z14.d, #0, #1
	mov z15.d, z14.d
	and z15.d, z15.d, #1
	cmpeq p3.d, p2/z, z15.d, #0
	dup z8.d, x24
	mul z15.d, p2/m, z15.d, z8.d
	mov z8.d, z15.d

	// C/bias scatter offsets for f32 and bf16 stores.
	index z14.s, #0, #1
	mov z15.d, z14.d
	and z15.s, z15.s, #3       // lane position inside 2x2 tile
	mov z9.d, z14.d
	lsr z9.s, z9.s, #2
	lsl z9.s, z9.s, #5         // segment * 8 cols * sizeof(f32)
	mov z13.d, z15.d
	and z13.s, z13.s, #1
	lsl z13.s, z13.s, #2       // col bit * sizeof(f32)
	add z9.s, z9.s, z13.s
	mov z10.d, z9.d            // bias offsets
	mov z12.d, z15.d
	lsr z12.s, z12.s, #1
	and z12.s, z12.s, #1       // row bit
	cmpeq p4.s, p1/z, z12.s, #0
	dup z13.s, w25
	mul z12.s, p1/m, z12.s, z13.s
	add z9.s, z9.s, z12.s

	mov z11.d, z14.d
	lsr z11.s, z11.s, #2
	lsl z11.s, z11.s, #4       // segment * 8 cols * sizeof(bf16)
	mov z13.d, z15.d
	and z13.s, z13.s, #1
	lsl z13.s, z13.s, #1       // col bit * sizeof(bf16)
	add z11.s, z11.s, z13.s
	mov z12.d, z15.d
	lsr z12.s, z12.s, #1
	and z12.s, z12.s, #1
	dup z13.s, w26
	mul z12.s, p1/m, z12.s, z13.s
	add z11.s, z11.s, z12.s

	mov x11, x1                // current B tile
	mov x12, x2                // current C tile
	mov x30, x5                // current bias tile, or NULL
	mov w10, w21               // remaining N columns

.L_n_loop:
	cmp w10, #0
	b.le .L_done

	LOAD_C_ALL_F32

	mov x16, x0                // current A k pointer
	mov x18, x28               // current packed-A k pointer
	mov x17, x11               // current B k pointer
	mov w15, w20               // remaining K

.L_k_loop_cached_entry:
	cbz x28, .L_k_loop_gather
	LOAD_AB_PACKED z0, z1, z2, z3, z4, z5, z6, z7
	subs w15, w15, #4
	b.eq .L_cached_tail_cur

.L_k_loop_cached:
	LOAD_AB_PACKED z8, z9, z10, z11, z12, z13, z14, z15
	BFMMLA_BLOCK_AB z0, z1, z2, z3, z4, z5, z6, z7
	subs w15, w15, #4
	b.eq .L_cached_tail_next
	LOAD_AB_PACKED z0, z1, z2, z3, z4, z5, z6, z7
	BFMMLA_BLOCK_AB z8, z9, z10, z11, z12, z13, z14, z15
	subs w15, w15, #4
	b.gt .L_k_loop_cached

.L_cached_tail_cur:
	BFMMLA_BLOCK_AB z0, z1, z2, z3, z4, z5, z6, z7
	b .L_k_done

.L_cached_tail_next:
	BFMMLA_BLOCK_AB z8, z9, z10, z11, z12, z13, z14, z15
	b .L_k_done

.L_k_loop_gather:
	ld1h {z4.h}, p0/z, [x17]
	add x3, x17, x27
	ld1h {z5.h}, p0/z, [x3]
	add x3, x3, x27
	ld1h {z6.h}, p0/z, [x3]
	add x3, x3, x27
	ld1h {z7.h}, p0/z, [x3]
	add x17, x17, x27, lsl #2

	cmp w19, #8
	b.ne .L_load_a_tail
	ld1d {z0.d}, p2/z, [x16, z8.d]
	add x3, x16, x24, lsl #1
	ld1d {z1.d}, p2/z, [x3, z8.d]
	add x3, x16, x24, lsl #2
	ld1d {z2.d}, p2/z, [x3, z8.d]
	add x3, x3, x24, lsl #1
	ld1d {z3.d}, p2/z, [x3, z8.d]
	b .L_compute

.L_load_a_tail:
	mov x3, x16
	LOAD_A_PAIR_TAIL z0, x3, 0
	add x3, x16, x24, lsl #1
	LOAD_A_PAIR_TAIL z1, x3, 2
	add x3, x16, x24, lsl #2
	LOAD_A_PAIR_TAIL z2, x3, 4
	add x3, x3, x24, lsl #1
	LOAD_A_PAIR_TAIL z3, x3, 6

.L_compute:
	BFMMLA_BLOCK
	add x16, x16, #8
	subs w15, w15, #4
	b.gt .L_k_loop_gather

.L_k_done:
	BUILD_C_OFFSETS
	mov x17, x12
	STORE_PAIR x17, 0, z16, z17, z18, z19
	cmp w7, #0
	b.ne 950f
	add x17, x12, x25, lsl #1
	b 951f
950:
	add x17, x12, x26, lsl #1
951:
	STORE_PAIR x17, 2, z20, z21, z22, z23
	cmp w7, #0
	b.ne 952f
	add x17, x12, x25, lsl #2
	b 953f
952:
	add x17, x12, x26, lsl #2
953:
	STORE_PAIR x17, 4, z24, z25, z26, z27
	cmp w7, #0
	b.ne 954f
	add x17, x17, x25, lsl #1
	b 955f
954:
	add x17, x17, x26, lsl #1
955:
	STORE_PAIR x17, 6, z28, z29, z30, z31

	add x11, x11, x13
	cmp w7, #0
	b.ne .L_advance_bf16
	add x12, x12, x14
	b .L_advance_common
.L_advance_bf16:
	add x12, x12, x27
.L_advance_common:
	cbz x30, .L_no_bias_advance
	add x30, x30, x14
.L_no_bias_advance:
	sub w10, w10, w9
	b .L_n_loop

.L_done:
	RESTORE_REGS
	ret

	.p2align 4
bf16gemm_sve_tail_m4_asm:
	SAVE_REGS

	ldp w19, w20, [x4]        // M, K
	ldr w21, [x4, #8]         // N
	ldr w23, [x4, #20]        // ldc
	mov x28, x3               // packed A
	cbz x28, .L_m4_fallback

	cntb x27
	lsr x9, x27, #1
	lsl x25, x23, #2
	lsl x26, x23, #1
	mul x13, x20, x27
	lsl x14, x27, #1
	ptrue p0.b
	ptrue p1.s

	mov x11, x1
	mov x12, x2
	mov x30, x5
	mov w10, w21

.L_m4_n_loop:
	cmp w10, #0
	b.le .L_m4_done

	ZERO_ACC_M4
	mov x17, x11
	mov x18, x28
	mov w15, w20

	LOAD_AB_PACKED_M4 z0, z1, z4, z5, z6, z7
	subs w15, w15, #4
	b.eq .L_m4_tail_cur

.L_m4_k_loop:
	LOAD_AB_PACKED_M4 z8, z9, z12, z13, z14, z15
	BFMMLA_M4_AB z0, z1, z4, z5, z6, z7
	subs w15, w15, #4
	b.eq .L_m4_tail_next
	LOAD_AB_PACKED_M4 z0, z1, z4, z5, z6, z7
	BFMMLA_M4_AB z8, z9, z12, z13, z14, z15
	subs w15, w15, #4
	b.gt .L_m4_k_loop

.L_m4_tail_cur:
	BFMMLA_M4_AB z0, z1, z4, z5, z6, z7
	b .L_m4_k_done

.L_m4_tail_next:
	BFMMLA_M4_AB z8, z9, z12, z13, z14, z15

.L_m4_k_done:
	BUILD_C_OFFSETS
	mov x17, x12
	STORE_PAIR x17, 0, z16, z17, z18, z19
	cmp w7, #0
	b.ne .L_m4_row2_bf16
	add x17, x12, x25, lsl #1
	b .L_m4_row2_store
.L_m4_row2_bf16:
	add x17, x12, x26, lsl #1
.L_m4_row2_store:
	STORE_PAIR x17, 2, z20, z21, z22, z23

	add x11, x11, x13
	cmp w7, #0
	b.ne .L_m4_advance_bf16
	add x12, x12, x14
	b .L_m4_advance_common
.L_m4_advance_bf16:
	add x12, x12, x27
.L_m4_advance_common:
	cbz x30, .L_m4_no_bias_advance
	add x30, x30, x14
.L_m4_no_bias_advance:
	sub w10, w10, w9
	b .L_m4_n_loop

.L_m4_done:
	RESTORE_REGS
	ret

.L_m4_fallback:
	RESTORE_REGS
	b bf16gemm_sve_kernel_asm

	.p2align 4
bf16gemm_sve_tail_m2_asm:
	SAVE_REGS

	ldp w19, w20, [x4]        // M, K
	ldr w21, [x4, #8]         // N
	ldr w23, [x4, #20]        // ldc
	mov x28, x3               // packed A
	cbz x28, .L_m2_fallback

	cntb x27
	lsr x9, x27, #1
	lsl x25, x23, #2
	lsl x26, x23, #1
	mul x13, x20, x27
	lsl x14, x27, #1
	ptrue p0.b
	ptrue p1.s

	mov x11, x1
	mov x12, x2
	mov x30, x5
	mov w10, w21

.L_m2_n_loop:
	cmp w10, #0
	b.le .L_m2_done

	ZERO_ACC_M2
	mov x17, x11
	mov x18, x28
	mov w15, w20

	LOAD_AB_PACKED_M2 z0, z4, z5, z6, z7
	subs w15, w15, #4
	b.eq .L_m2_tail_cur

.L_m2_k_loop:
	LOAD_AB_PACKED_M2 z8, z12, z13, z14, z15
	BFMMLA_M2_AB z0, z4, z5, z6, z7
	subs w15, w15, #4
	b.eq .L_m2_tail_next
	LOAD_AB_PACKED_M2 z0, z4, z5, z6, z7
	BFMMLA_M2_AB z8, z12, z13, z14, z15
	subs w15, w15, #4
	b.gt .L_m2_k_loop

.L_m2_tail_cur:
	BFMMLA_M2_AB z0, z4, z5, z6, z7
	b .L_m2_k_done

.L_m2_tail_next:
	BFMMLA_M2_AB z8, z12, z13, z14, z15

.L_m2_k_done:
	BUILD_C_OFFSETS
	mov x17, x12
	STORE_PAIR x17, 0, z16, z17, z18, z19

	add x11, x11, x13
	cmp w7, #0
	b.ne .L_m2_advance_bf16
	add x12, x12, x14
	b .L_m2_advance_common
.L_m2_advance_bf16:
	add x12, x12, x27
.L_m2_advance_common:
	cbz x30, .L_m2_no_bias_advance
	add x30, x30, x14
.L_m2_no_bias_advance:
	sub w10, w10, w9
	b .L_m2_n_loop

.L_m2_done:
	RESTORE_REGS
	ret

.L_m2_fallback:
	RESTORE_REGS
	b bf16gemm_sve_kernel_asm

	.global bf16gemm_k_nld_f_m12
	.type bf16gemm_k_nld_f_m12, %function
	.p2align 4
bf16gemm_k_nld_f_m12:
	SAVE_REGS

	ldr w20, [x4, #4]         // K
	ldr w21, [x4, #8]         // N
	ldr w23, [x4, #20]        // ldc
	mov x28, x3               // packed A base

	cntb x27
	lsr x9, x27, #1
	lsl x25, x23, #2
	mul x13, x20, x27
	lsl x14, x27, #1
	ptrue p0.b
	ptrue p1.s

	mov x11, x1
	mov x12, x2
	mov w10, w21

.L_m12_n_loop:
	cmp w10, #0
	b.le .L_m12_done

	ZERO_ACC_M12
	mov x17, x11
	mov x18, x28
	mov w15, w20

.L_m12_k_loop:
	M12_K4_BODY
	M12_K4_BODY
	subs w15, w15, #8
	b.gt .L_m12_k_loop

	BUILD_F32_OFFSETS_M12
	mov x16, x12
	STORE_PAIR_M12 x16, z8, z9, z10, z11
	add x16, x12, x25, lsl #1
	STORE_PAIR_M12 x16, z12, z13, z14, z15
	add x16, x12, x25, lsl #2
	STORE_PAIR_M12 x16, z16, z17, z18, z19
	add x16, x16, x25, lsl #1
	STORE_PAIR_M12 x16, z20, z21, z22, z23
	add x16, x12, x25, lsl #3
	STORE_PAIR_M12 x16, z24, z25, z26, z27
	add x16, x16, x25, lsl #1
	STORE_PAIR_M12 x16, z28, z29, z30, z31

	add x11, x11, x13
	add x12, x12, x14
	sub w10, w10, w9
	b .L_m12_n_loop

.L_m12_done:
	RESTORE_REGS
	ret
	.size bf16gemm_k_nld_f_m12, .-bf16gemm_k_nld_f_m12

	.macro ENTRY_NO_BIAS name, loadc, storebf16
	.global \name
	.type \name, %function
\name:
	mov x5, #0
	mov w6, #\loadc
	mov w7, #\storebf16
	b bf16gemm_sve_kernel_asm
	.size \name, .-\name
	.endm

	.macro ENTRY_NO_BIAS_TO name, target, loadc, storebf16
	.global \name
	.type \name, %function
\name:
	mov x5, #0
	mov w6, #\loadc
	mov w7, #\storebf16
	b \target
	.size \name, .-\name
	.endm

	.macro ENTRY_BIAS name
	.global \name
	.type \name, %function
\name:
	mov w6, #0
	mov w7, #0
	b bf16gemm_sve_kernel_asm
	.size \name, .-\name
	.endm

	.macro ENTRY_BIAS_TO name, target
	.global \name
	.type \name, %function
\name:
	mov w6, #0
	mov w7, #0
	b \target
	.size \name, .-\name
	.endm

	ENTRY_NO_BIAS bf16gemm_k_ld,     0, 0
	ENTRY_NO_BIAS_TO bf16gemm_k_ld1, bf16gemm_sve_tail_m2_asm, 0, 0
	ENTRY_NO_BIAS_TO bf16gemm_k_ld2, bf16gemm_sve_tail_m2_asm, 0, 0
	ENTRY_NO_BIAS_TO bf16gemm_k_ld4, bf16gemm_sve_tail_m4_asm, 0, 0
	ENTRY_NO_BIAS bf16gemm_k_ld_b,   0, 1
	ENTRY_NO_BIAS_TO bf16gemm_k_ld1_b, bf16gemm_sve_tail_m2_asm, 0, 1
	ENTRY_NO_BIAS_TO bf16gemm_k_ld2_b, bf16gemm_sve_tail_m2_asm, 0, 1
	ENTRY_NO_BIAS_TO bf16gemm_k_ld4_b, bf16gemm_sve_tail_m4_asm, 0, 1
	ENTRY_NO_BIAS bf16gemm_k_nld_b,  0, 1
	ENTRY_NO_BIAS_TO bf16gemm_k_nld1_b, bf16gemm_sve_tail_m2_asm, 0, 1
	ENTRY_NO_BIAS_TO bf16gemm_k_nld2_b, bf16gemm_sve_tail_m2_asm, 0, 1
	ENTRY_NO_BIAS_TO bf16gemm_k_nld4_b, bf16gemm_sve_tail_m4_asm, 0, 1

	ENTRY_BIAS bf16gemm_k_ld_bias_f
	ENTRY_BIAS_TO bf16gemm_k_ld1_bias_f, bf16gemm_sve_tail_m2_asm
	ENTRY_BIAS_TO bf16gemm_k_ld2_bias_f, bf16gemm_sve_tail_m2_asm
	ENTRY_BIAS_TO bf16gemm_k_ld4_bias_f, bf16gemm_sve_tail_m4_asm

	.section .note.GNU-stack,"",@progbits
