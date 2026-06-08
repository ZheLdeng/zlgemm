// bf16gemm_sve_nld.S -- compute-only SVE BF16 loop probe.
//
// void bf16gemm_sve_nld_compute(uint64_t k4_iters)
//
// Each iteration executes the same 12xNtile K4 BFMMLA schedule as
// bf16gemm_k_nld_f_m12, but removes all A/B loads and C stores.

	.text
	.align 4

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

	.macro M12_K4_COMPUTE
	BFMMLA_M12_2ROWS z0, z1, z8, z9, z10, z11, z12, z13, z14, z15
	BFMMLA_M12_2ROWS z2, z3, z16, z17, z18, z19, z20, z21, z22, z23
	BFMMLA_M12_2A
	.endm

	.global bf16gemm_sve_nld_compute
	.type bf16gemm_sve_nld_compute, %function
	.p2align 4
bf16gemm_sve_nld_compute:
	stp d8,  d9,  [sp, #-16]!
	stp d10, d11, [sp, #-16]!
	stp d12, d13, [sp, #-16]!
	stp d14, d15, [sp, #-16]!
	cbz x0, .L_done
	dup z0.h, #1
	dup z1.h, #2
	dup z2.h, #3
	dup z3.h, #4
	dup z4.h, #5
	dup z5.h, #6
	dup z6.h, #7
	dup z7.h, #8
	ZERO_ACC_M12

.L_loop:
	M12_K4_COMPUTE
	subs x0, x0, #1
	b.ne .L_loop

.L_done:
	ldp d14, d15, [sp], #16
	ldp d12, d13, [sp], #16
	ldp d10, d11, [sp], #16
	ldp d8,  d9,  [sp], #16
	ret
	.size bf16gemm_sve_nld_compute, .-bf16gemm_sve_nld_compute

	.section .note.GNU-stack,"",@progbits
