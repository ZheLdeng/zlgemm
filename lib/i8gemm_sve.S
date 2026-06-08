// i8gemm_sve.S -- SVE I8 GEMM kernel entry points.
//
// ABI matches the NEON low-level kernels:
//   x0 A, x1 packed B, x2 C, x3 packed A or NULL, x4 params
// Bias variants additionally receive x5 = bias.

	.text
	.arch armv8.6-a+sve+i8mm

	.macro SAVE_REGS
	stp x19, x20, [sp, #-16]!
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
	ldp x19, x20, [sp], #16
	.endm

	// Load two A rows (8 int8 each) into one replicated SVE vector.
	// row pair index is an immediate 0..3. Uses x7/x18 as temporaries.
	.macro LOAD_A_PAIR tag, zreg, rp
	mov x7, #(\rp * 2)
	cmp x7, x3
	b.ge .L_\tag\()_load_a_pair_\rp\()_zero
	add x7, x7, x15
	madd x7, x7, x11, x4
	mov x18, #(\rp * 2 + 1)
	cmp x18, x3
	b.ge .L_\tag\()_load_a_pair_\rp\()_zero_high
	add x18, x18, x15
	madd x18, x18, x11, x4
	ldr d0, [x0, x7]
	add x18, x0, x18
	ld1 {v0.d}[1], [x18]
	b .L_\tag\()_load_a_pair_\rp\()_dup
.L_\tag\()_load_a_pair_\rp\()_zero_high:
	ldr d0, [x0, x7]
	mov v0.d[1], xzr
.L_\tag\()_load_a_pair_\rp\()_dup:
	dup \zreg\().q, z0.q[0]
	b .L_\tag\()_load_a_pair_\rp\()_done
.L_\tag\()_load_a_pair_\rp\()_zero:
	mov \zreg\().b, #0
.L_\tag\()_load_a_pair_\rp\()_done:
	.endm

	.macro MMLA_ROW areg, c0, c1, c2, c3
	smmla \c0\().s, \areg\().b, z4.b
	smmla \c1\().s, \areg\().b, z5.b
	smmla \c2\().s, \areg\().b, z6.b
	smmla \c3\().s, \areg\().b, z7.b
	.endm

	.macro MMLA_BLOCK a0, a1, a2, a3, b0, b1, b2, b3
	smmla z16.s, \a0\().b, \b0\().b
	smmla z17.s, \a0\().b, \b1\().b
	smmla z18.s, \a0\().b, \b2\().b
	smmla z19.s, \a0\().b, \b3\().b
	smmla z20.s, \a1\().b, \b0\().b
	smmla z21.s, \a1\().b, \b1\().b
	smmla z22.s, \a1\().b, \b2\().b
	smmla z23.s, \a1\().b, \b3\().b
	smmla z24.s, \a2\().b, \b0\().b
	smmla z25.s, \a2\().b, \b1\().b
	smmla z26.s, \a2\().b, \b2\().b
	smmla z27.s, \a2\().b, \b3\().b
	smmla z28.s, \a3\().b, \b0\().b
	smmla z29.s, \a3\().b, \b1\().b
	smmla z30.s, \a3\().b, \b2\().b
	smmla z31.s, \a3\().b, \b3\().b
	.endm

	.macro LOAD_AB_PACKED_CUR
	ld1b z4.b, p0/z, [x6]
	ld1b z5.b, p0/z, [x6, #1, mul vl]
	ld1b z6.b, p0/z, [x6, #2, mul vl]
	ld1b z7.b, p0/z, [x6, #3, mul vl]
	addvl x6, x6, #4
	ld1rqb z0.b, p0/z, [x20]
	ld1rqb z1.b, p0/z, [x20, #16]
	ld1rqb z2.b, p0/z, [x20, #32]
	ld1rqb z3.b, p0/z, [x20, #48]
	add x20, x20, #64
	.endm

	.macro LOAD_AB_PACKED_NEXT
	ld1b z12.b, p0/z, [x6]
	ld1b z13.b, p0/z, [x6, #1, mul vl]
	ld1b z14.b, p0/z, [x6, #2, mul vl]
	ld1b z15.b, p0/z, [x6, #3, mul vl]
	addvl x6, x6, #4
	ld1rqb z8.b, p0/z, [x20]
	ld1rqb z9.b, p0/z, [x20, #16]
	ld1rqb z10.b, p0/z, [x20, #32]
	ld1rqb z11.b, p0/z, [x20, #48]
	add x20, x20, #64
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

	.macro MMLA_M12_4A
	smmla z8.s,  z0.b, z4.b
	smmla z12.s, z1.b, z4.b
	smmla z16.s, z2.b, z4.b
	smmla z20.s, z3.b, z4.b
	smmla z9.s,  z0.b, z5.b
	smmla z13.s, z1.b, z5.b
	smmla z17.s, z2.b, z5.b
	smmla z21.s, z3.b, z5.b
	smmla z10.s, z0.b, z6.b
	smmla z14.s, z1.b, z6.b
	smmla z18.s, z2.b, z6.b
	smmla z22.s, z3.b, z6.b
	smmla z11.s, z0.b, z7.b
	smmla z15.s, z1.b, z7.b
	smmla z19.s, z2.b, z7.b
	smmla z23.s, z3.b, z7.b
	.endm

	.macro MMLA_M12_2A
	smmla z24.s, z0.b, z4.b
	smmla z28.s, z1.b, z4.b
	smmla z25.s, z0.b, z5.b
	smmla z29.s, z1.b, z5.b
	smmla z26.s, z0.b, z6.b
	smmla z30.s, z1.b, z6.b
	smmla z27.s, z0.b, z7.b
	smmla z31.s, z1.b, z7.b
	.endm

	.macro M12_K8_BODY
	ld1b z4.b, p0/z, [x6]
	ld1b z5.b, p0/z, [x6, #1, mul vl]
	ld1b z6.b, p0/z, [x6, #2, mul vl]
	ld1b z7.b, p0/z, [x6, #3, mul vl]
	addvl x6, x6, #4
	ld1rqb z0.b, p0/z, [x20]
	ld1rqb z1.b, p0/z, [x20, #16]
	ld1rqb z2.b, p0/z, [x20, #32]
	ld1rqb z3.b, p0/z, [x20, #48]
	MMLA_M12_4A
	ld1rqb z0.b, p0/z, [x20, #64]
	ld1rqb z1.b, p0/z, [x20, #80]
	MMLA_M12_2A
	add x20, x20, #96
	.endm

	// Build element offsets for one column pair:
	// lane pattern per 128b segment:
	//   r0/c0, r0/c1, r1/c0, r1/c1
	.macro BUILD_OFFSETS cp
	index z0.s, #0, #1
	lsr z1.s, z0.s, #2
	lsl z1.s, z1.s, #3
	mov z2.d, z0.d
	and z2.s, z2.s, #1
	add z1.s, z1.s, z2.s
	add z1.s, z1.s, #(\cp * 2)
	mov z2.d, z0.d
	and z2.s, z2.s, #2
	lsr z2.s, z2.s, #1
	dup z3.s, w12
	mul z2.s, p1/m, z2.s, z3.s
	add z2.s, z2.s, z1.s
	.endm

	.macro BUILD_I32_OFFSET_BASE out
	index \out\().s, #0, #1
	lsr \out\().s, \out\().s, #2
	lsl \out\().s, \out\().s, #3
	index z1.s, #0, #1
	and z1.s, z1.s, #1
	add \out\().s, \out\().s, z1.s
	index z1.s, #0, #1
	and z1.s, z1.s, #2
	lsr z1.s, z1.s, #1
	dup z0.s, w12
	mul z1.s, p1/m, z1.s, z0.s
	add \out\().s, \out\().s, z1.s
	.endm

	.macro BUILD_I32_STORE_OFFSETS
	BUILD_I32_OFFSET_BASE z8
	mov z9.d, z8.d
	add z9.s, z9.s, #2
	mov z10.d, z8.d
	add z10.s, z10.s, #4
	mov z11.d, z8.d
	add z11.s, z11.s, #6
	.endm

	.macro BUILD_I32_STORE_OFFSETS_M12
	BUILD_I32_OFFSET_BASE z0
	mov z1.d, z0.d
	add z1.s, z1.s, #2
	mov z2.d, z0.d
	add z2.s, z2.s, #4
	mov z3.d, z0.d
	add z3.s, z3.s, #6
	.endm

	// Build column offsets used by f32 stores once per N tile.
	// z8-z11: C offsets for cp0..cp3, z12-z15: bias offsets for cp0..cp3.
	.macro BUILD_F32_STORE_OFFSETS
	index z8.s, #0, #1
	lsr z12.s, z8.s, #2
	lsl z12.s, z12.s, #3
	mov z13.d, z8.d
	and z13.s, z13.s, #1
	add z12.s, z12.s, z13.s
	mov z13.d, z8.d
	and z13.s, z13.s, #2
	lsr z13.s, z13.s, #1
	dup z14.s, w12
	mul z13.s, p1/m, z13.s, z14.s
	add z8.s, z13.s, z12.s
	mov z9.d, z8.d
	add z9.s, z9.s, #2
	mov z10.d, z8.d
	add z10.s, z10.s, #4
	mov z11.d, z8.d
	add z11.s, z11.s, #6
	mov z13.d, z12.d
	add z13.s, z13.s, #2
	mov z14.d, z12.d
	add z14.s, z14.s, #4
	mov z15.d, z12.d
	add z15.s, z15.s, #6
	.endm

	.macro LOAD_C_PAIR acc, cp, pred
	BUILD_OFFSETS \cp
	ld1w \acc\().s, \pred/z, [x18, z2.s, sxtw #2]
	.endm

	.macro STORE_PAIR_I32 acc, cp, pred
	BUILD_OFFSETS \cp
	st1w \acc\().s, \pred, [x18, z2.s, sxtw #2]
	.endm

	.macro STORE_PAIR_F32_OFF acc, off, pred, tmp
	scvtf \tmp\().s, p1/m, \acc\().s
	st1w \tmp\().s, \pred, [x18, \off\().s, sxtw #2]
	.endm

	.macro STORE_PAIR_F32_BIAS_OFF acc, coff, boff, pred, tmp, btmp
	scvtf \tmp\().s, p1/m, \acc\().s
	ld1w \btmp\().s, p1/z, [x6, \boff\().s, sxtw #2]
	fadd \tmp\().s, \tmp\().s, \btmp\().s
	st1w \tmp\().s, \pred, [x18, \coff\().s, sxtw #2]
	.endm

	.macro LOAD_C_ROW tag, rp, a0, a1, a2, a3
	mov x4, #(\rp * 8)
	madd x18, x4, x12, x18
	cmp x3, #(\rp * 2)
	b.le .L_\tag\()_load_c_row_\rp\()_zero
	cmp x3, #(\rp * 2 + 1)
	b.le .L_\tag\()_load_c_row_\rp\()_one
	LOAD_C_PAIR \a0, 0, p1
	LOAD_C_PAIR \a1, 1, p1
	LOAD_C_PAIR \a2, 2, p1
	LOAD_C_PAIR \a3, 3, p1
	b .L_\tag\()_load_c_row_\rp\()_done
.L_\tag\()_load_c_row_\rp\()_one:
	LOAD_C_PAIR \a0, 0, p4
	LOAD_C_PAIR \a1, 1, p4
	LOAD_C_PAIR \a2, 2, p4
	LOAD_C_PAIR \a3, 3, p4
	b .L_\tag\()_load_c_row_\rp\()_done
.L_\tag\()_load_c_row_\rp\()_zero:
	mov \a0\().s, #0
	mov \a1\().s, #0
	mov \a2\().s, #0
	mov \a3\().s, #0
.L_\tag\()_load_c_row_\rp\()_done:
	mov x4, #(\rp * 8)
	msub x18, x4, x12, x18
	.endm

	.macro STORE_ROW_I32 tag, rp, a0, a1, a2, a3
	mov x4, #(\rp * 8)
	madd x18, x4, x12, x18
	cmp x3, #(\rp * 2)
	b.le .L_\tag\()_store_i32_row_\rp\()_done
	cmp x3, #(\rp * 2 + 1)
	b.le .L_\tag\()_store_i32_row_\rp\()_one
	STORE_PAIR_I32 \a0, 0, p1
	STORE_PAIR_I32 \a1, 1, p1
	STORE_PAIR_I32 \a2, 2, p1
	STORE_PAIR_I32 \a3, 3, p1
	b .L_\tag\()_store_i32_row_\rp\()_done
.L_\tag\()_store_i32_row_\rp\()_one:
	STORE_PAIR_I32 \a0, 0, p4
	STORE_PAIR_I32 \a1, 1, p4
	STORE_PAIR_I32 \a2, 2, p4
	STORE_PAIR_I32 \a3, 3, p4
.L_\tag\()_store_i32_row_\rp\()_done:
	mov x4, #(\rp * 8)
	msub x18, x4, x12, x18
	.endm

	.macro STORE_ROW_F32 tag, rp, a0, a1, a2, a3
	mov x4, #(\rp * 8)
	madd x18, x4, x12, x18
	cmp x3, #(\rp * 2)
	b.le .L_\tag\()_store_f32_row_\rp\()_done
	cmp x3, #(\rp * 2 + 1)
	b.le .L_\tag\()_store_f32_row_\rp\()_one
	STORE_PAIR_F32_OFF \a0, z8, p1, z4
	STORE_PAIR_F32_OFF \a1, z9, p1, z5
	STORE_PAIR_F32_OFF \a2, z10, p1, z6
	STORE_PAIR_F32_OFF \a3, z11, p1, z7
	b .L_\tag\()_store_f32_row_\rp\()_done
.L_\tag\()_store_f32_row_\rp\()_one:
	STORE_PAIR_F32_OFF \a0, z8, p4, z4
	STORE_PAIR_F32_OFF \a1, z9, p4, z5
	STORE_PAIR_F32_OFF \a2, z10, p4, z6
	STORE_PAIR_F32_OFF \a3, z11, p4, z7
.L_\tag\()_store_f32_row_\rp\()_done:
	mov x4, #(\rp * 8)
	msub x18, x4, x12, x18
	.endm

	.macro STORE_ROW_F32_BIAS tag, rp, a0, a1, a2, a3
	mov x4, #(\rp * 8)
	madd x18, x4, x12, x18
	cmp x3, #(\rp * 2)
	b.le .L_\tag\()_store_bias_row_\rp\()_done
	cmp x3, #(\rp * 2 + 1)
	b.le .L_\tag\()_store_bias_row_\rp\()_one
	STORE_PAIR_F32_BIAS_OFF \a0, z8, z12, p1, z4, z0
	STORE_PAIR_F32_BIAS_OFF \a1, z9, z13, p1, z5, z1
	STORE_PAIR_F32_BIAS_OFF \a2, z10, z14, p1, z6, z2
	STORE_PAIR_F32_BIAS_OFF \a3, z11, z15, p1, z7, z3
	b .L_\tag\()_store_bias_row_\rp\()_done
.L_\tag\()_store_bias_row_\rp\()_one:
	STORE_PAIR_F32_BIAS_OFF \a0, z8, z12, p4, z4, z0
	STORE_PAIR_F32_BIAS_OFF \a1, z9, z13, p4, z5, z1
	STORE_PAIR_F32_BIAS_OFF \a2, z10, z14, p4, z6, z2
	STORE_PAIR_F32_BIAS_OFF \a3, z11, z15, p4, z7, z3
.L_\tag\()_store_bias_row_\rp\()_done:
	mov x4, #(\rp * 8)
	msub x18, x4, x12, x18
	.endm

	.macro STORE_ROW_I32_FAST tag, rp, a0, a1, a2, a3
	mov x4, #(\rp * 8)
	madd x18, x4, x12, x18
	cmp x3, #(\rp * 2)
	b.le .L_\tag\()_store_i32_fast_row_\rp\()_done
	cmp x3, #(\rp * 2 + 1)
	b.le .L_\tag\()_store_i32_fast_row_\rp\()_one
	st1w \a0\().s, p1, [x18, z8.s, sxtw #2]
	st1w \a1\().s, p1, [x18, z9.s, sxtw #2]
	st1w \a2\().s, p1, [x18, z10.s, sxtw #2]
	st1w \a3\().s, p1, [x18, z11.s, sxtw #2]
	b .L_\tag\()_store_i32_fast_row_\rp\()_done
.L_\tag\()_store_i32_fast_row_\rp\()_one:
	st1w \a0\().s, p4, [x18, z8.s, sxtw #2]
	st1w \a1\().s, p4, [x18, z9.s, sxtw #2]
	st1w \a2\().s, p4, [x18, z10.s, sxtw #2]
	st1w \a3\().s, p4, [x18, z11.s, sxtw #2]
.L_\tag\()_store_i32_fast_row_\rp\()_done:
	mov x4, #(\rp * 8)
	msub x18, x4, x12, x18
	.endm

	.macro STORE_ROW_I32_M12 rp, a0, a1, a2, a3
	mov x4, #(\rp * 8)
	madd x18, x4, x12, x18
	cmp x8, #(\rp * 2)
	b.le .L_store_i32_m12_row_\rp\()_done
	cmp x8, #(\rp * 2 + 1)
	b.le .L_store_i32_m12_row_\rp\()_one
	st1w \a0\().s, p1, [x18, z0.s, sxtw #2]
	st1w \a1\().s, p1, [x18, z1.s, sxtw #2]
	st1w \a2\().s, p1, [x18, z2.s, sxtw #2]
	st1w \a3\().s, p1, [x18, z3.s, sxtw #2]
	b .L_store_i32_m12_row_\rp\()_done
.L_store_i32_m12_row_\rp\()_one:
	st1w \a0\().s, p4, [x18, z0.s, sxtw #2]
	st1w \a1\().s, p4, [x18, z1.s, sxtw #2]
	st1w \a2\().s, p4, [x18, z2.s, sxtw #2]
	st1w \a3\().s, p4, [x18, z3.s, sxtw #2]
.L_store_i32_m12_row_\rp\()_done:
	mov x4, #(\rp * 8)
	msub x18, x4, x12, x18
	.endm

	.global i8gemm_k_nld_m12
	.type i8gemm_k_nld_m12, %function
i8gemm_k_nld_m12:
	SAVE_REGS
	mov x19, x3               // packed-A base

	ldr w8, [x4, #0]          // M, normally <= 12
	ldr w9, [x4, #4]          // K
	ldr w10, [x4, #8]         // N
	ldr w12, [x4, #20]        // ldc

	ptrue p0.b
	ptrue p1.s
	index z0.s, #0, #1
	and z0.s, z0.s, #2
	cmpeq p4.s, p1/z, z0.s, #0

	cntb x13
	lsr x13, x13, #1         // n_tile = VL bytes / 2
	mul x14, x9, x13         // packed-B N-tile stride in bytes

	mov x16, #0              // n0
	mov x17, x1              // current B tile
.L_m12_n_loop:
	mov x18, x16
	add x18, x2, x18, lsl #2
	ZERO_ACC_M12

	mov x4, x9
	mov x6, x17
	mov x20, x19
.L_m12_k_loop:
	M12_K8_BODY
	M12_K8_BODY
	subs x4, x4, #16
	b.gt .L_m12_k_loop

	BUILD_I32_STORE_OFFSETS_M12
	mov x18, x16
	add x18, x2, x18, lsl #2
	STORE_ROW_I32_M12 0, z8, z9, z10, z11
	STORE_ROW_I32_M12 1, z12, z13, z14, z15
	STORE_ROW_I32_M12 2, z16, z17, z18, z19
	STORE_ROW_I32_M12 3, z20, z21, z22, z23
	STORE_ROW_I32_M12 4, z24, z25, z26, z27
	STORE_ROW_I32_M12 5, z28, z29, z30, z31

	add x16, x16, x13
	add x17, x17, x14
	cmp x16, x10
	b.lt .L_m12_n_loop

	RESTORE_REGS
	ret
	.size i8gemm_k_nld_m12, .-i8gemm_k_nld_m12

	.macro LOAD_C_ACC tag
	LOAD_C_ROW \tag, 0, z16, z17, z18, z19
	LOAD_C_ROW \tag, 1, z20, z21, z22, z23
	LOAD_C_ROW \tag, 2, z24, z25, z26, z27
	LOAD_C_ROW \tag, 3, z28, z29, z30, z31
	.endm

	.macro LOAD_C_ZERO tag
	ZERO_ACC
	.endm

	.macro STORE_C_I32 tag
	BUILD_I32_STORE_OFFSETS
	STORE_ROW_I32_FAST \tag, 0, z16, z17, z18, z19
	STORE_ROW_I32_FAST \tag, 1, z20, z21, z22, z23
	STORE_ROW_I32_FAST \tag, 2, z24, z25, z26, z27
	STORE_ROW_I32_FAST \tag, 3, z28, z29, z30, z31
	.endm

	.macro STORE_C_F32 tag
	BUILD_F32_STORE_OFFSETS
	STORE_ROW_F32 \tag, 0, z16, z17, z18, z19
	STORE_ROW_F32 \tag, 1, z20, z21, z22, z23
	STORE_ROW_F32 \tag, 2, z24, z25, z26, z27
	STORE_ROW_F32 \tag, 3, z28, z29, z30, z31
	.endm

	.macro STORE_C_F32_BIAS tag
	add x6, x5, x16, lsl #2
	BUILD_F32_STORE_OFFSETS
	STORE_ROW_F32_BIAS \tag, 0, z16, z17, z18, z19
	STORE_ROW_F32_BIAS \tag, 1, z20, z21, z22, z23
	STORE_ROW_F32_BIAS \tag, 2, z24, z25, z26, z27
	STORE_ROW_F32_BIAS \tag, 3, z28, z29, z30, z31
	.endm

	.macro GEMM_BODY func_name, load_c, store_c
	.global \func_name
	.type \func_name, %function
\func_name:
	SAVE_REGS
	mov x19, x3               // optional packed-A base, NULL for gather path

	ldr w8, [x4, #0]          // M
	ldr w9, [x4, #4]          // K
	ldr w10, [x4, #8]         // N
	ldr w11, [x4, #12]        // lda
	ldr w12, [x4, #20]        // ldc

	ptrue p0.b
	ptrue p1.s

	index z0.s, #0, #1
	and z0.s, z0.s, #2
	cmpeq p4.s, p1/z, z0.s, #0

	cntb x13
	lsr x13, x13, #1         // n_tile = VL bytes / 2
	mul x14, x9, x13         // B packed tile stride in bytes

	mov x15, #0              // m0
.L_\func_name\()_m_loop:
	cmp x15, x8
	b.ge .L_\func_name\()_done
	sub x3, x8, x15
	cmp x3, #8
	mov x18, #8
	csel x3, x3, x18, le     // m_now = min(M - m0, 8)

	mov x16, #0              // n0
	mov x17, x1              // current B tile
.L_\func_name\()_n_loop:
	// C base for this M/N tile, in bytes.
	madd x18, x15, x12, x16
	add x18, x2, x18, lsl #2

	\load_c \func_name

	cbnz x19, .L_\func_name\()_k_loop_packed_entry
	mov x4, #0               // kb
	mov x6, x17              // B K pointer
.L_\func_name\()_k_loop:
	cmp x4, x9
	b.ge .L_\func_name\()_store

	ld1b z4.b, p0/z, [x6]
	ld1b z5.b, p0/z, [x6, #1, mul vl]
	ld1b z6.b, p0/z, [x6, #2, mul vl]
	ld1b z7.b, p0/z, [x6, #3, mul vl]

	LOAD_A_PAIR \func_name, z0, 0
	MMLA_ROW z0, z16, z17, z18, z19
	LOAD_A_PAIR \func_name, z0, 1
	MMLA_ROW z0, z20, z21, z22, z23
	LOAD_A_PAIR \func_name, z0, 2
	MMLA_ROW z0, z24, z25, z26, z27
	LOAD_A_PAIR \func_name, z0, 3
	MMLA_ROW z0, z28, z29, z30, z31

	add x4, x4, #8
	addvl x6, x6, #4
	b .L_\func_name\()_k_loop

.L_\func_name\()_k_loop_packed_entry:
	mov x4, #0
	mov x6, x17              // B K pointer
	lsr x20, x15, #3
	mul x20, x20, x9
	add x20, x19, x20, lsl #3

	cmp x4, x9
	b.ge .L_\func_name\()_store
	LOAD_AB_PACKED_CUR
	add x4, x4, #8
	cmp x4, x9
	b.ge .L_\func_name\()_packed_tail_cur

.L_\func_name\()_k_loop_packed:
	LOAD_AB_PACKED_NEXT
	MMLA_BLOCK z0, z1, z2, z3, z4, z5, z6, z7
	add x4, x4, #8
	cmp x4, x9
	b.ge .L_\func_name\()_packed_tail_next
	LOAD_AB_PACKED_CUR
	MMLA_BLOCK z8, z9, z10, z11, z12, z13, z14, z15
	add x4, x4, #8
	cmp x4, x9
	b.lt .L_\func_name\()_k_loop_packed

.L_\func_name\()_packed_tail_cur:
	MMLA_BLOCK z0, z1, z2, z3, z4, z5, z6, z7
	b .L_\func_name\()_store

.L_\func_name\()_packed_tail_next:
	MMLA_BLOCK z8, z9, z10, z11, z12, z13, z14, z15
	b .L_\func_name\()_store

.L_\func_name\()_store:
	madd x18, x15, x12, x16
	add x18, x2, x18, lsl #2
	\store_c \func_name

	add x16, x16, x13
	add x17, x17, x14
	cmp x16, x10
	b.lt .L_\func_name\()_n_loop

	add x15, x15, #8
	b .L_\func_name\()_m_loop

.L_\func_name\()_done:
	RESTORE_REGS
	ret
	.size \func_name, .-\func_name
	.endm

	GEMM_BODY i8gemm_k_ld, LOAD_C_ACC, STORE_C_I32
	GEMM_BODY i8gemm_k_nld, LOAD_C_ZERO, STORE_C_I32
	GEMM_BODY i8gemm_k_ld1, LOAD_C_ACC, STORE_C_I32
	GEMM_BODY i8gemm_k_nld1, LOAD_C_ZERO, STORE_C_I32
	GEMM_BODY i8gemm_k_ld2, LOAD_C_ACC, STORE_C_I32
	GEMM_BODY i8gemm_k_nld2, LOAD_C_ZERO, STORE_C_I32
	GEMM_BODY i8gemm_k_ld4, LOAD_C_ACC, STORE_C_I32
	GEMM_BODY i8gemm_k_nld4, LOAD_C_ZERO, STORE_C_I32

	GEMM_BODY i8gemm_k_ld_f, LOAD_C_ZERO, STORE_C_F32
	GEMM_BODY i8gemm_k_ld1_f, LOAD_C_ZERO, STORE_C_F32
	GEMM_BODY i8gemm_k_ld2_f, LOAD_C_ZERO, STORE_C_F32
	GEMM_BODY i8gemm_k_ld4_f, LOAD_C_ZERO, STORE_C_F32

	GEMM_BODY i8gemm_k_ld_bias_f, LOAD_C_ZERO, STORE_C_F32_BIAS
	GEMM_BODY i8gemm_k_ld1_bias_f, LOAD_C_ZERO, STORE_C_F32_BIAS
	GEMM_BODY i8gemm_k_ld2_bias_f, LOAD_C_ZERO, STORE_C_F32_BIAS
	GEMM_BODY i8gemm_k_ld4_bias_f, LOAD_C_ZERO, STORE_C_F32_BIAS
