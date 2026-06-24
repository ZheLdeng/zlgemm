// huawei_roofline.S -- pure-compute smmla roofline, accumulator-count sweep.
//
// Goal (Huawei 80-core): find the smmla issue-rate ceiling and whether it is
// latency-bound (rises with #accumulators) or throughput-capped (flat). Each
// routine runs a tight loop of N *independent* smmla (distinct accumulators,
// one shared operand pair) with NO memory access, so the only limit is the
// SIMD issue/latency of smmla itself.
//
// SVE smmla (VL=256) = 64 MAC = 128 2-ops/instr.
// NEON smmla (128b)  = 32 MAC = 64  2-ops/instr.
// void f(uint64_t iters);  x0 = iters.

	.text
	.arch armv8.6-a+sve+i8mm

	.macro SAVE_FP
	stp d8,  d9,  [sp, #-16]!
	stp d10, d11, [sp, #-16]!
	stp d12, d13, [sp, #-16]!
	stp d14, d15, [sp, #-16]!
	.endm
	.macro REST_FP
	ldp d14, d15, [sp], #16
	ldp d12, d13, [sp], #16
	ldp d10, d11, [sp], #16
	ldp d8,  d9,  [sp], #16
	.endm

// ---- SVE, 8 accumulators (z24..z31), operands z0,z1 ----
	.global roof_sve_8
	.type roof_sve_8,%function
roof_sve_8:
	mov z0.b, #1
	mov z1.b, #1
	mov z24.s,#0
	mov z25.s,#0
	mov z26.s,#0
	mov z27.s,#0
	mov z28.s,#0
	mov z29.s,#0
	mov z30.s,#0
	mov z31.s,#0
1:	smmla z24.s, z0.b, z1.b
	smmla z25.s, z0.b, z1.b
	smmla z26.s, z0.b, z1.b
	smmla z27.s, z0.b, z1.b
	smmla z28.s, z0.b, z1.b
	smmla z29.s, z0.b, z1.b
	smmla z30.s, z0.b, z1.b
	smmla z31.s, z0.b, z1.b
	subs x0, x0, #1
	b.gt 1b
	ret
	.size roof_sve_8,.-roof_sve_8

// ---- SVE, 16 accumulators (z16..z31) ----
	.global roof_sve_16
	.type roof_sve_16,%function
roof_sve_16:
	mov z0.b,#1
	mov z1.b,#1
	mov z16.s,#0
	mov z17.s,#0
	mov z18.s,#0
	mov z19.s,#0
	mov z20.s,#0
	mov z21.s,#0
	mov z22.s,#0
	mov z23.s,#0
	mov z24.s,#0
	mov z25.s,#0
	mov z26.s,#0
	mov z27.s,#0
	mov z28.s,#0
	mov z29.s,#0
	mov z30.s,#0
	mov z31.s,#0
1:	smmla z16.s, z0.b, z1.b
	smmla z17.s, z0.b, z1.b
	smmla z18.s, z0.b, z1.b
	smmla z19.s, z0.b, z1.b
	smmla z20.s, z0.b, z1.b
	smmla z21.s, z0.b, z1.b
	smmla z22.s, z0.b, z1.b
	smmla z23.s, z0.b, z1.b
	smmla z24.s, z0.b, z1.b
	smmla z25.s, z0.b, z1.b
	smmla z26.s, z0.b, z1.b
	smmla z27.s, z0.b, z1.b
	smmla z28.s, z0.b, z1.b
	smmla z29.s, z0.b, z1.b
	smmla z30.s, z0.b, z1.b
	smmla z31.s, z0.b, z1.b
	subs x0, x0, #1
	b.gt 1b
	ret
	.size roof_sve_16,.-roof_sve_16

// ---- SVE, 24 accumulators (z8..z31) ----
	.global roof_sve_24
	.type roof_sve_24,%function
roof_sve_24:
	SAVE_FP
	mov z0.b,#1
	mov z1.b,#1
	mov z8.s,#0
	mov z9.s,#0
	mov z10.s,#0
	mov z11.s,#0
	mov z12.s,#0
	mov z13.s,#0
	mov z14.s,#0
	mov z15.s,#0
	mov z16.s,#0
	mov z17.s,#0
	mov z18.s,#0
	mov z19.s,#0
	mov z20.s,#0
	mov z21.s,#0
	mov z22.s,#0
	mov z23.s,#0
	mov z24.s,#0
	mov z25.s,#0
	mov z26.s,#0
	mov z27.s,#0
	mov z28.s,#0
	mov z29.s,#0
	mov z30.s,#0
	mov z31.s,#0
1:	smmla z8.s,  z0.b, z1.b
	smmla z9.s,  z0.b, z1.b
	smmla z10.s, z0.b, z1.b
	smmla z11.s, z0.b, z1.b
	smmla z12.s, z0.b, z1.b
	smmla z13.s, z0.b, z1.b
	smmla z14.s, z0.b, z1.b
	smmla z15.s, z0.b, z1.b
	smmla z16.s, z0.b, z1.b
	smmla z17.s, z0.b, z1.b
	smmla z18.s, z0.b, z1.b
	smmla z19.s, z0.b, z1.b
	smmla z20.s, z0.b, z1.b
	smmla z21.s, z0.b, z1.b
	smmla z22.s, z0.b, z1.b
	smmla z23.s, z0.b, z1.b
	smmla z24.s, z0.b, z1.b
	smmla z25.s, z0.b, z1.b
	smmla z26.s, z0.b, z1.b
	smmla z27.s, z0.b, z1.b
	smmla z28.s, z0.b, z1.b
	smmla z29.s, z0.b, z1.b
	smmla z30.s, z0.b, z1.b
	smmla z31.s, z0.b, z1.b
	subs x0, x0, #1
	b.gt 1b
	REST_FP
	ret
	.size roof_sve_24,.-roof_sve_24

// ---- SVE, 30 accumulators (z2..z31), operands z0,z1 ----
	.global roof_sve_30
	.type roof_sve_30,%function
roof_sve_30:
	SAVE_FP
	mov z0.b,#1
	mov z1.b,#1
	.irp r,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
	mov z\r\().s,#0
	.endr
1:
	.irp r,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
	smmla z\r\().s, z0.b, z1.b
	.endr
	subs x0, x0, #1
	b.gt 1b
	REST_FP
	ret
	.size roof_sve_30,.-roof_sve_30

// ---- NEON reference, 24 accumulators (v8..v31), operands v0,v1 ----
	.global roof_neon_24
	.type roof_neon_24,%function
roof_neon_24:
	SAVE_FP
	movi v0.16b, #1
	movi v1.16b, #1
	.irp r,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
	movi v\r\().4s, #0
	.endr
1:
	.irp r,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
	smmla v\r\().4s, v0.16b, v1.16b
	.endr
	subs x0, x0, #1
	b.gt 1b
	REST_FP
	ret
	.size roof_neon_24,.-roof_neon_24

	.section .note.GNU-stack,"",@progbits
