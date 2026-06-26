// huawei_bf16_roofline.S -- pure-compute bfmmla roofline, accumulator sweep.
//
// Mirrors huawei_roofline.S (smmla) for bf16. Each routine runs a tight loop of
// N *independent* bfmmla (distinct accumulators, fixed register inputs, NO
// memory) to expose the bfmmla SIMD issue/latency ceiling. If GOPS rises with
// the accumulator count it is latency-bound; if flat it is throughput-capped.
//
// SVE bfmmla (VL=256) = 2 segments x (2x2x4) = 32 MAC = 64 2-ops/instr.
//   x0 = iteration count. bf16 inputs set to 1.0 (0x3F80) to avoid any zero/
//   denormal fast-path skewing the throughput measurement.

	.arch armv8.6-a+sve+bf16
	.text

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

	// roof_bf16_8: 8 independent accumulators (z24..z31)
	.global roof_bf16_8
	.type roof_bf16_8, %function
roof_bf16_8:
	SAVE_FP
	mov w9, #0x3F80
	dup z0.h, w9
	dup z1.h, w9
	mov z24.s,#0
	mov z25.s,#0
	mov z26.s,#0
	mov z27.s,#0
	mov z28.s,#0
	mov z29.s,#0
	mov z30.s,#0
	mov z31.s,#0
1:	bfmmla z24.s, z0.h, z1.h
	bfmmla z25.s, z0.h, z1.h
	bfmmla z26.s, z0.h, z1.h
	bfmmla z27.s, z0.h, z1.h
	bfmmla z28.s, z0.h, z1.h
	bfmmla z29.s, z0.h, z1.h
	bfmmla z30.s, z0.h, z1.h
	bfmmla z31.s, z0.h, z1.h
	subs x0, x0, #1
	b.ne 1b
	REST_FP
	ret
	.size roof_bf16_8, .-roof_bf16_8

	// roof_bf16_16: 16 accumulators (z16..z31)
	.global roof_bf16_16
	.type roof_bf16_16, %function
roof_bf16_16:
	SAVE_FP
	mov w9, #0x3F80
	dup z0.h, w9
	dup z1.h, w9
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
1:	bfmmla z16.s, z0.h, z1.h
	bfmmla z17.s, z0.h, z1.h
	bfmmla z18.s, z0.h, z1.h
	bfmmla z19.s, z0.h, z1.h
	bfmmla z20.s, z0.h, z1.h
	bfmmla z21.s, z0.h, z1.h
	bfmmla z22.s, z0.h, z1.h
	bfmmla z23.s, z0.h, z1.h
	bfmmla z24.s, z0.h, z1.h
	bfmmla z25.s, z0.h, z1.h
	bfmmla z26.s, z0.h, z1.h
	bfmmla z27.s, z0.h, z1.h
	bfmmla z28.s, z0.h, z1.h
	bfmmla z29.s, z0.h, z1.h
	bfmmla z30.s, z0.h, z1.h
	bfmmla z31.s, z0.h, z1.h
	subs x0, x0, #1
	b.ne 1b
	REST_FP
	ret
	.size roof_bf16_16, .-roof_bf16_16

	// roof_bf16_24: 24 accumulators (z8..z31), z2/z3 as inputs
	.global roof_bf16_24
	.type roof_bf16_24, %function
roof_bf16_24:
	SAVE_FP
	mov w9, #0x3F80
	dup z2.h, w9
	dup z3.h, w9
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
1:	bfmmla z8.s,  z2.h, z3.h
	bfmmla z9.s,  z2.h, z3.h
	bfmmla z10.s, z2.h, z3.h
	bfmmla z11.s, z2.h, z3.h
	bfmmla z12.s, z2.h, z3.h
	bfmmla z13.s, z2.h, z3.h
	bfmmla z14.s, z2.h, z3.h
	bfmmla z15.s, z2.h, z3.h
	bfmmla z16.s, z2.h, z3.h
	bfmmla z17.s, z2.h, z3.h
	bfmmla z18.s, z2.h, z3.h
	bfmmla z19.s, z2.h, z3.h
	bfmmla z20.s, z2.h, z3.h
	bfmmla z21.s, z2.h, z3.h
	bfmmla z22.s, z2.h, z3.h
	bfmmla z23.s, z2.h, z3.h
	bfmmla z24.s, z2.h, z3.h
	bfmmla z25.s, z2.h, z3.h
	bfmmla z26.s, z2.h, z3.h
	bfmmla z27.s, z2.h, z3.h
	bfmmla z28.s, z2.h, z3.h
	bfmmla z29.s, z2.h, z3.h
	bfmmla z30.s, z2.h, z3.h
	bfmmla z31.s, z2.h, z3.h
	subs x0, x0, #1
	b.ne 1b
	REST_FP
	ret
	.size roof_bf16_24, .-roof_bf16_24

	// roof_bf16_30: 30 accumulators (z2..z31), z0/z1 inputs
	.global roof_bf16_30
	.type roof_bf16_30, %function
roof_bf16_30:
	SAVE_FP
	mov w9, #0x3F80
	dup z0.h, w9
	dup z1.h, w9
	mov z2.s,#0
	mov z3.s,#0
	mov z4.s,#0
	mov z5.s,#0
	mov z6.s,#0
	mov z7.s,#0
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
1:	bfmmla z2.s,  z0.h, z1.h
	bfmmla z3.s,  z0.h, z1.h
	bfmmla z4.s,  z0.h, z1.h
	bfmmla z5.s,  z0.h, z1.h
	bfmmla z6.s,  z0.h, z1.h
	bfmmla z7.s,  z0.h, z1.h
	bfmmla z8.s,  z0.h, z1.h
	bfmmla z9.s,  z0.h, z1.h
	bfmmla z10.s, z0.h, z1.h
	bfmmla z11.s, z0.h, z1.h
	bfmmla z12.s, z0.h, z1.h
	bfmmla z13.s, z0.h, z1.h
	bfmmla z14.s, z0.h, z1.h
	bfmmla z15.s, z0.h, z1.h
	bfmmla z16.s, z0.h, z1.h
	bfmmla z17.s, z0.h, z1.h
	bfmmla z18.s, z0.h, z1.h
	bfmmla z19.s, z0.h, z1.h
	bfmmla z20.s, z0.h, z1.h
	bfmmla z21.s, z0.h, z1.h
	bfmmla z22.s, z0.h, z1.h
	bfmmla z23.s, z0.h, z1.h
	bfmmla z24.s, z0.h, z1.h
	bfmmla z25.s, z0.h, z1.h
	bfmmla z26.s, z0.h, z1.h
	bfmmla z27.s, z0.h, z1.h
	bfmmla z28.s, z0.h, z1.h
	bfmmla z29.s, z0.h, z1.h
	bfmmla z30.s, z0.h, z1.h
	bfmmla z31.s, z0.h, z1.h
	subs x0, x0, #1
	b.ne 1b
	REST_FP
	ret
	.size roof_bf16_30, .-roof_bf16_30

	.section .note.GNU-stack,"",@progbits
