// i8gemm_sve.S -- SVE I8 GEMM kernel entry points.
//
// ABI matches the NEON low-level kernels:
//   x0 A, x1 packed B, x2 C, x3 A_reorder (unused), x4 params
// Bias variants additionally receive x5 = bias.

	.text
	.arch armv8.6-a+sve+i8mm

	.macro ENTRY_I32 name, loadc
	.global \name
	.type \name, %function
\name:
	mov x5, xzr
	mov w6, #\loadc
	mov w7, #0
	b i8gemm_sve_common
	.size \name, .-\name
	.endm

	.macro ENTRY_F32 name
	.global \name
	.type \name, %function
\name:
	mov x5, xzr
	mov w6, #0
	mov w7, #1
	b i8gemm_sve_common
	.size \name, .-\name
	.endm

	.macro ENTRY_BIAS name
	.global \name
	.type \name, %function
\name:
	mov w6, #0
	mov w7, #1
	b i8gemm_sve_common
	.size \name, .-\name
	.endm

	// Load two A rows (8 int8 each) into one replicated SVE vector.
	// row pair index is an immediate 0..3. Uses x18 and x3 as temporaries.
	.macro LOAD_A_PAIR zreg, rp
	mov x18, #(\rp * 2)
	cmp x18, x3
	b.ge 92\@f
	add x18, x18, x15
	madd x18, x18, x11, x4
	ldr d0, [x0, x18]
	mov x18, #(\rp * 2 + 1)
	cmp x18, x3
	b.ge 91\@f
	add x18, x18, x15
	madd x18, x18, x11, x4
	add x18, x0, x18
	ld1 {v0.d}[1], [x18]
	b 90\@f
91\@:
	mov v0.d[1], xzr
90\@:
	dup \zreg\().q, z0.q[0]
	b 93\@f
92\@:
	mov \zreg\().b, #0
93\@:
	.endm

	.macro MMLA_ROW areg, c0, c1, c2, c3
	smmla \c0\().s, \areg\().b, z4.b
	smmla \c1\().s, \areg\().b, z5.b
	smmla \c2\().s, \areg\().b, z6.b
	smmla \c3\().s, \areg\().b, z7.b
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

	.macro LOAD_C_PAIR acc, cp, pred
	BUILD_OFFSETS \cp
	ld1w \acc\().s, \pred/z, [x18, z2.s, sxtw #2]
	.endm

	.macro STORE_PAIR acc, cp, pred
	BUILD_OFFSETS \cp
	cbz w7, 95\@f
	scvtf z3.s, p1/m, \acc\().s
	cbz x5, 94\@f
	ld1w z0.s, p1/z, [x6, z1.s, sxtw #2]
	fadd z3.s, z3.s, z0.s
94\@:
	st1w z3.s, \pred, [x18, z2.s, sxtw #2]
	b 96\@f
95\@:
	st1w \acc\().s, \pred, [x18, z2.s, sxtw #2]
96\@:
	.endm

	.macro LOAD_C_ROW rp, a0, a1, a2, a3
	mov x4, #(\rp * 8)
	madd x18, x4, x12, x18
	cmp x3, #(\rp * 2)
	b.le 97\@f
	cmp x3, #(\rp * 2 + 1)
	b.le 98\@f
	LOAD_C_PAIR \a0, 0, p1
	LOAD_C_PAIR \a1, 1, p1
	LOAD_C_PAIR \a2, 2, p1
	LOAD_C_PAIR \a3, 3, p1
	b 99\@f
98\@:
	LOAD_C_PAIR \a0, 0, p4
	LOAD_C_PAIR \a1, 1, p4
	LOAD_C_PAIR \a2, 2, p4
	LOAD_C_PAIR \a3, 3, p4
	b 99\@f
97\@:
	mov \a0\().s, #0
	mov \a1\().s, #0
	mov \a2\().s, #0
	mov \a3\().s, #0
99\@:
	mov x4, #(\rp * 8)
	msub x18, x4, x12, x18
	.endm

	.macro STORE_ROW rp, a0, a1, a2, a3
	mov x4, #(\rp * 8)
	madd x18, x4, x12, x18
	cmp x3, #(\rp * 2)
	b.le 97\@f
	cmp x3, #(\rp * 2 + 1)
	b.le 98\@f
	STORE_PAIR \a0, 0, p1
	STORE_PAIR \a1, 1, p1
	STORE_PAIR \a2, 2, p1
	STORE_PAIR \a3, 3, p1
	b 97\@f
98\@:
	STORE_PAIR \a0, 0, p4
	STORE_PAIR \a1, 1, p4
	STORE_PAIR \a2, 2, p4
	STORE_PAIR \a3, 3, p4
97\@:
	mov x4, #(\rp * 8)
	msub x18, x4, x12, x18
	.endm

	.type i8gemm_sve_common, %function
i8gemm_sve_common:
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
.L_m_loop:
	cmp x15, x8
	b.ge .L_done
	sub x3, x8, x15
	cmp x3, #8
	mov x18, #8
	csel x3, x3, x18, le     // m_now = min(M - m0, 8)

	mov x16, #0              // n0
	mov x17, x1              // current B tile
.L_n_loop:
	// C base for this M/N tile, in bytes.
	madd x18, x15, x12, x16
	add x18, x2, x18, lsl #2

	cbz w6, .L_zero_acc
	LOAD_C_ROW 0, z16, z17, z18, z19
	LOAD_C_ROW 1, z20, z21, z22, z23
	LOAD_C_ROW 2, z24, z25, z26, z27
	LOAD_C_ROW 3, z28, z29, z30, z31
	b .L_k_start
.L_zero_acc:
	ZERO_ACC

.L_k_start:
	mov x4, #0               // kb
	mov x6, x17              // B K pointer
.L_k_loop:
	cmp x4, x9
	b.ge .L_store

	ld1b z4.b, p0/z, [x6]
	ld1b z5.b, p0/z, [x6, #1, mul vl]
	ld1b z6.b, p0/z, [x6, #2, mul vl]
	ld1b z7.b, p0/z, [x6, #3, mul vl]

	LOAD_A_PAIR z0, 0
	MMLA_ROW z0, z16, z17, z18, z19
	LOAD_A_PAIR z0, 1
	MMLA_ROW z0, z20, z21, z22, z23
	LOAD_A_PAIR z0, 2
	MMLA_ROW z0, z24, z25, z26, z27
	LOAD_A_PAIR z0, 3
	MMLA_ROW z0, z28, z29, z30, z31

	add x4, x4, #8
	addvl x6, x6, #4
	b .L_k_loop

.L_store:
	madd x18, x15, x12, x16
	add x18, x2, x18, lsl #2
	add x6, x5, x16, lsl #2
	STORE_ROW 0, z16, z17, z18, z19
	STORE_ROW 1, z20, z21, z22, z23
	STORE_ROW 2, z24, z25, z26, z27
	STORE_ROW 3, z28, z29, z30, z31

	add x16, x16, x13
	add x17, x17, x14
	cmp x16, x10
	b.lt .L_n_loop

	add x15, x15, #8
	b .L_m_loop

.L_done:
	ret
	.size i8gemm_sve_common, .-i8gemm_sve_common

	ENTRY_I32 i8gemm_k_ld, 1
	ENTRY_I32 i8gemm_k_ld1, 1
	ENTRY_I32 i8gemm_k_ld2, 1
	ENTRY_I32 i8gemm_k_ld4, 1

	ENTRY_F32 i8gemm_k_ld_f
	ENTRY_F32 i8gemm_k_ld1_f
	ENTRY_F32 i8gemm_k_ld2_f
	ENTRY_F32 i8gemm_k_ld4_f

	ENTRY_BIAS i8gemm_k_ld_bias_f
	ENTRY_BIAS i8gemm_k_ld1_bias_f
	ENTRY_BIAS i8gemm_k_ld2_bias_f
	ENTRY_BIAS i8gemm_k_ld4_bias_f
