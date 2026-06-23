// compute_ceiling.S -- pure-compute and compute+load ceiling microkernels.
//
// Each routine runs a matmul tile in a tight loop. The *_only_* variants do NO
// loads (operands resident) and expose the pure instruction-issue ceiling.
// The *_ld_* variants add the realistic operand loads from a tiny L1-resident
// cyclic buffer (x1) so we can isolate how much the loads cost.
// x0 = iteration count, x1 = base of a >=128B aligned scratch buffer.

	.text
	.arch armv8.6-a+sve+i8mm+bf16

// ---- pure compute, 16 smmla (4x4 i8 tile) ----
	.global compute_only_i8_16
	.type compute_only_i8_16, %function
compute_only_i8_16:
	mov z0.b, #1
	mov z1.b, #1
	mov z2.b, #1
	mov z3.b, #1
	mov z4.b, #1
	mov z5.b, #1
	mov z6.b, #1
	mov z7.b, #1
	ptrue p0.b
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
1:
	smmla z16.s, z0.b, z4.b
	smmla z17.s, z0.b, z5.b
	smmla z18.s, z0.b, z6.b
	smmla z19.s, z0.b, z7.b
	smmla z20.s, z1.b, z4.b
	smmla z21.s, z1.b, z5.b
	smmla z22.s, z1.b, z6.b
	smmla z23.s, z1.b, z7.b
	smmla z24.s, z2.b, z4.b
	smmla z25.s, z2.b, z5.b
	smmla z26.s, z2.b, z6.b
	smmla z27.s, z2.b, z7.b
	smmla z28.s, z3.b, z4.b
	smmla z29.s, z3.b, z5.b
	smmla z30.s, z3.b, z6.b
	smmla z31.s, z3.b, z7.b
	subs x0, x0, #1
	b.gt 1b
	ret
	.size compute_only_i8_16, .-compute_only_i8_16

// ---- compute + load, 16 smmla + 4 ld1b(B) + 4 ld1rqb(A), intensity 2.0 ----
// Mirrors the 4x4 double-buffered main kernel's per-K8 load:compute ratio.
	.global compute_ld_i8_16
	.type compute_ld_i8_16, %function
compute_ld_i8_16:
	ptrue p0.b
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
1:
	ld1b  z4.b, p0/z, [x1]
	ld1b  z5.b, p0/z, [x1, #1, mul vl]
	ld1b  z6.b, p0/z, [x1, #2, mul vl]
	ld1b  z7.b, p0/z, [x1, #3, mul vl]
	ld1rqb z0.b, p0/z, [x1]
	ld1rqb z1.b, p0/z, [x1, #16]
	ld1rqb z2.b, p0/z, [x1, #32]
	ld1rqb z3.b, p0/z, [x1, #48]
	smmla z16.s, z0.b, z4.b
	smmla z17.s, z0.b, z5.b
	smmla z18.s, z0.b, z6.b
	smmla z19.s, z0.b, z7.b
	smmla z20.s, z1.b, z4.b
	smmla z21.s, z1.b, z5.b
	smmla z22.s, z1.b, z6.b
	smmla z23.s, z1.b, z7.b
	smmla z24.s, z2.b, z4.b
	smmla z25.s, z2.b, z5.b
	smmla z26.s, z2.b, z6.b
	smmla z27.s, z2.b, z7.b
	smmla z28.s, z3.b, z4.b
	smmla z29.s, z3.b, z5.b
	smmla z30.s, z3.b, z6.b
	smmla z31.s, z3.b, z7.b
	subs x0, x0, #1
	b.gt 1b
	ret
	.size compute_ld_i8_16, .-compute_ld_i8_16

// ---- compute + load, 24 smmla + 4 ld1b(B) + 6 ld1rqb(A), intensity 2.4 ----
// Mirrors the m12 / ACL 8x3VL load:compute ratio (24 acc, no double buffer).
	.global compute_ld_i8_24
	.type compute_ld_i8_24, %function
compute_ld_i8_24:
	ptrue p0.b
	stp d8,  d9,  [sp, #-16]!
	stp d10, d11, [sp, #-16]!
	stp d12, d13, [sp, #-16]!
	stp d14, d15, [sp, #-16]!
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
1:
	ld1b  z4.b, p0/z, [x1]
	ld1b  z5.b, p0/z, [x1, #1, mul vl]
	ld1b  z6.b, p0/z, [x1, #2, mul vl]
	ld1b  z7.b, p0/z, [x1, #3, mul vl]
	ld1rqb z0.b, p0/z, [x1]
	ld1rqb z1.b, p0/z, [x1, #16]
	ld1rqb z2.b, p0/z, [x1, #32]
	smmla z8.s,  z0.b, z4.b
	smmla z9.s,  z0.b, z5.b
	smmla z10.s, z0.b, z6.b
	smmla z11.s, z0.b, z7.b
	smmla z12.s, z1.b, z4.b
	smmla z13.s, z1.b, z5.b
	smmla z14.s, z1.b, z6.b
	smmla z15.s, z1.b, z7.b
	smmla z16.s, z2.b, z4.b
	smmla z17.s, z2.b, z5.b
	smmla z18.s, z2.b, z6.b
	smmla z19.s, z2.b, z7.b
	ld1rqb z0.b, p0/z, [x1, #64]
	ld1rqb z1.b, p0/z, [x1, #80]
	ld1rqb z2.b, p0/z, [x1, #96]
	smmla z20.s, z0.b, z4.b
	smmla z21.s, z0.b, z5.b
	smmla z22.s, z0.b, z6.b
	smmla z23.s, z0.b, z7.b
	smmla z24.s, z1.b, z4.b
	smmla z25.s, z1.b, z5.b
	smmla z26.s, z1.b, z6.b
	smmla z27.s, z1.b, z7.b
	smmla z28.s, z2.b, z4.b
	smmla z29.s, z2.b, z5.b
	smmla z30.s, z2.b, z6.b
	smmla z31.s, z2.b, z7.b
	subs x0, x0, #1
	b.gt 1b
	ldp d14, d15, [sp], #16
	ldp d12, d13, [sp], #16
	ldp d10, d11, [sp], #16
	ldp d8,  d9,  [sp], #16
	ret
	.size compute_ld_i8_24, .-compute_ld_i8_24

// ---- pure compute, 16 bfmmla ----
	.global compute_only_bf16_16
	.type compute_only_bf16_16, %function
compute_only_bf16_16:
	mov z0.h, #1
	mov z1.h, #1
	mov z2.h, #1
	mov z3.h, #1
	mov z4.h, #1
	mov z5.h, #1
	mov z6.h, #1
	mov z7.h, #1
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
1:
	bfmmla z16.s, z0.h, z4.h
	bfmmla z17.s, z0.h, z5.h
	bfmmla z18.s, z0.h, z6.h
	bfmmla z19.s, z0.h, z7.h
	bfmmla z20.s, z1.h, z4.h
	bfmmla z21.s, z1.h, z5.h
	bfmmla z22.s, z1.h, z6.h
	bfmmla z23.s, z1.h, z7.h
	bfmmla z24.s, z2.h, z4.h
	bfmmla z25.s, z2.h, z5.h
	bfmmla z26.s, z2.h, z6.h
	bfmmla z27.s, z2.h, z7.h
	bfmmla z28.s, z3.h, z4.h
	bfmmla z29.s, z3.h, z5.h
	bfmmla z30.s, z3.h, z6.h
	bfmmla z31.s, z3.h, z7.h
	subs x0, x0, #1
	b.gt 1b
	ret
	.size compute_only_bf16_16, .-compute_only_bf16_16

	.section .note.GNU-stack,"",@progbits
