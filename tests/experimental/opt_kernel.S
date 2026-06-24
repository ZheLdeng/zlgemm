// opt_kernel.S -- experimental optimized i8 8x16 (M8 x N=2VL) microkernel.
//
// Goal: validate that >=90% pre-packed utilization is reachable by keeping the
// double-buffered CUR/NEXT load scheduling and asm style, but cutting per-step
// overhead the way ACL does:
//   * unroll the K loop to K16 (two K8 steps) per branch
//   * address B/A with immediate `mul vl` / byte offsets within the unrolled block
//   * advance the base pointers only once per K16 block
//
// void i8_kernel_opt(const int8_t *A_pack, const int8_t *B_pack,
//                    int32_t *C, int K);   // M=8, N=16, K multiple of 16
//
// Packed layouts match i8_pack_A_neon_m8_asm (A) and i8_pack_B with n_tile=16.

	.text
	.arch armv8.6-a+sve+i8mm

	.global i8_kernel_opt
	.type i8_kernel_opt, %function
i8_kernel_opt:
	ptrue p0.b
	stp d8,  d9,  [sp, #-16]!
	stp d10, d11, [sp, #-16]!
	stp d12, d13, [sp, #-16]!
	stp d14, d15, [sp, #-16]!
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

	mov x4, x3                 // K counter (down)
	// prologue: load CUR (K8 #0): B z4-7 from [x1], A z0-3 from [x0]
	ld1b z4.b, p0/z, [x1]
	ld1b z5.b, p0/z, [x1, #1, mul vl]
	ld1b z6.b, p0/z, [x1, #2, mul vl]
	ld1b z7.b, p0/z, [x1, #3, mul vl]
	ld1rqb z0.b, p0/z, [x0]
	ld1rqb z1.b, p0/z, [x0, #16]
	ld1rqb z2.b, p0/z, [x0, #32]
	ld1rqb z3.b, p0/z, [x0, #48]

.Lopt_k:
	// ---- CUR step: compute z0-7, load NEXT (K8 #1) into z8-15 via offsets ----
	smmla z16.s, z0.b, z4.b
	smmla z17.s, z0.b, z5.b
	ld1b z12.b, p0/z, [x1, #4, mul vl]
	smmla z18.s, z0.b, z6.b
	smmla z19.s, z0.b, z7.b
	ld1b z13.b, p0/z, [x1, #5, mul vl]
	smmla z20.s, z1.b, z4.b
	smmla z21.s, z1.b, z5.b
	ld1b z14.b, p0/z, [x1, #6, mul vl]
	smmla z22.s, z1.b, z6.b
	smmla z23.s, z1.b, z7.b
	ld1b z15.b, p0/z, [x1, #7, mul vl]
	smmla z24.s, z2.b, z4.b
	smmla z25.s, z2.b, z5.b
	ld1rqb z8.b, p0/z, [x0, #64]
	smmla z26.s, z2.b, z6.b
	smmla z27.s, z2.b, z7.b
	ld1rqb z9.b, p0/z, [x0, #80]
	smmla z28.s, z3.b, z4.b
	smmla z29.s, z3.b, z5.b
	ld1rqb z10.b, p0/z, [x0, #96]
	smmla z30.s, z3.b, z6.b
	smmla z31.s, z3.b, z7.b
	ld1rqb z11.b, p0/z, [x0, #112]

	// advance base pointers once per K16 block
	addvl x1, x1, #8
	add x0, x0, #128

	// ---- NEXT step: compute z8-15, load CUR (next block K8 #0) into z0-7 ----
	smmla z16.s, z8.b, z12.b
	smmla z17.s, z8.b, z13.b
	ld1b z4.b, p0/z, [x1]
	smmla z18.s, z8.b, z14.b
	smmla z19.s, z8.b, z15.b
	ld1b z5.b, p0/z, [x1, #1, mul vl]
	smmla z20.s, z9.b, z12.b
	smmla z21.s, z9.b, z13.b
	ld1b z6.b, p0/z, [x1, #2, mul vl]
	smmla z22.s, z9.b, z14.b
	smmla z23.s, z9.b, z15.b
	ld1b z7.b, p0/z, [x1, #3, mul vl]
	smmla z24.s, z10.b, z12.b
	smmla z25.s, z10.b, z13.b
	ld1rqb z0.b, p0/z, [x0]
	smmla z26.s, z10.b, z14.b
	smmla z27.s, z10.b, z15.b
	ld1rqb z1.b, p0/z, [x0, #16]
	smmla z28.s, z11.b, z12.b
	smmla z29.s, z11.b, z13.b
	ld1rqb z2.b, p0/z, [x0, #32]
	smmla z30.s, z11.b, z14.b
	smmla z31.s, z11.b, z15.b
	ld1rqb z3.b, p0/z, [x0, #48]

	subs x4, x4, #16
	b.gt .Lopt_k

	// simple (not layout-correct) store, amortized -- timing only
	st1w z16.s, p0, [x2]
	st1w z17.s, p0, [x2, #1, mul vl]
	st1w z18.s, p0, [x2, #2, mul vl]
	st1w z19.s, p0, [x2, #3, mul vl]
	ldp d14, d15, [sp], #16
	ldp d12, d13, [sp], #16
	ldp d10, d11, [sp], #16
	ldp d8,  d9,  [sp], #16
	ret
	.size i8_kernel_opt, .-i8_kernel_opt

	.section .note.GNU-stack,"",@progbits
