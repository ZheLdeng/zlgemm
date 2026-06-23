// i8gemm_m16n4.S -- NEON I8MM 16x4 kernel for packed-A m16n4 path.
//
// void i8gemm_k_m16n4_packed_asm(const int8_t *A_pack,   // x0
//                                const int8_t *B_reo,    // x1
//                                int32_t      *C,        // x2, row-major
//                                int K_r,                // w3
//                                int N_r,                // w4, C ldc
//                                int n0,                 // w5
//                                int rows,               // w6, 1..16
//                                int zero_c);            // w7
//
// Contract:
//   - Computes one 16x4 output block; stores only the first rows rows.
//   - A_pack layout matches i8_pack_A_m16n4_block in i8gemm_m16n4.c:
//       for each K8 block:
//         8 q-registers, each containing two rows x 8 bytes.
//   - B_reo layout is the existing i8_pack_B 8-column format.
//   - C is int32 row-major with ldc=N_r; n0 selects the 4 output columns.

	.text
	.align 4

#ifdef __APPLE__
	.global _i8gemm_k_m16n4_packed_asm
_i8gemm_k_m16n4_packed_asm:
#else
	.global i8gemm_k_m16n4_packed_asm
	.type i8gemm_k_m16n4_packed_asm, %function
i8gemm_k_m16n4_packed_asm:
#endif
	stp d8,  d9,  [sp, #-16]!
	stp d10, d11, [sp, #-16]!
	stp d12, d13, [sp, #-16]!
	stp d14, d15, [sp, #-16]!

	// x0 = A_pack, x1 = B_reo, x2 = C
	// w3 = K_r, w4 = N_r, w5 = n0, w6 = rows, w7 = zero_c
	sxtw x3, w3
	sxtw x4, w4
	sxtw x5, w5
	sxtw x6, w6

	// C_N_BASE = C + n0 * sizeof(int32)
	add  x11, x2, x5, lsl #2
	// LDC_B = N_r * sizeof(int32)
	lsl  x10, x4, #2

	// B panel pointer:
	// B + (n0 / 8) * K_r * 8 + (n0 & 7) * 8
	lsr  x7, x5, #3
	mul  x8, x7, x3
	lsl  x8, x8, #3
	and  x9, x5, #7
	lsl  x9, x9, #3
	add  x1, x1, x8
	add  x1, x1, x9

	cbnz w7, .Lm16n4_zero_c

	// Load full 16x4 C block into blocked row-pair accumulators:
	// v16/v17 = rows 0/1, cols 0..1 / 2..3, etc.
	.macro LOAD_C_PAIR acc0, acc1
	ldr  q0, [x11]
	add  x12, x11, x10
	ldr  q1, [x12]
	zip1 \acc0\().2d, v0.2d, v1.2d
	zip2 \acc1\().2d, v0.2d, v1.2d
	add  x11, x11, x10, lsl #1
	.endm

	LOAD_C_PAIR v16, v17
	LOAD_C_PAIR v18, v19
	LOAD_C_PAIR v20, v21
	LOAD_C_PAIR v22, v23
	LOAD_C_PAIR v24, v25
	LOAD_C_PAIR v26, v27
	LOAD_C_PAIR v28, v29
	LOAD_C_PAIR v30, v31
	b .Lm16n4_k_loop_setup

.Lm16n4_zero_c:
	movi v16.4s, #0
	movi v17.4s, #0
	movi v18.4s, #0
	movi v19.4s, #0
	movi v20.4s, #0
	movi v21.4s, #0
	movi v22.4s, #0
	movi v23.4s, #0
	movi v24.4s, #0
	movi v25.4s, #0
	movi v26.4s, #0
	movi v27.4s, #0
	movi v28.4s, #0
	movi v29.4s, #0
	movi v30.4s, #0
	movi v31.4s, #0

.Lm16n4_k_loop_setup:
	lsr  x14, x3, #3
	cbz  x14, .Lm16n4_store
	lsr  x15, x14, #2
	and  x14, x14, #3
	cbz  x15, .Lm16n4_k_tail_loop

	.macro COMPUTE_M16N4 a0, a1, a2, a3, a4, a5, a6, a7, b0, b1
	smmla v16.4s, \a0\().16b, \b0\().16b
	smmla v17.4s, \a0\().16b, \b1\().16b

	smmla v18.4s, \a1\().16b, \b0\().16b
	smmla v19.4s, \a1\().16b, \b1\().16b

	smmla v20.4s, \a2\().16b, \b0\().16b
	smmla v21.4s, \a2\().16b, \b1\().16b

	smmla v22.4s, \a3\().16b, \b0\().16b
	smmla v23.4s, \a3\().16b, \b1\().16b

	smmla v24.4s, \a4\().16b, \b0\().16b
	smmla v25.4s, \a4\().16b, \b1\().16b

	smmla v26.4s, \a5\().16b, \b0\().16b
	smmla v27.4s, \a5\().16b, \b1\().16b

	smmla v28.4s, \a6\().16b, \b0\().16b
	smmla v29.4s, \a6\().16b, \b1\().16b

	smmla v30.4s, \a7\().16b, \b0\().16b
	smmla v31.4s, \a7\().16b, \b1\().16b
	.endm

	.macro LOAD_SLICE b0, b1, a0, a1, a2, a3, a4, a5, a6, a7
	ldr  \b0, [x1]
	ldr  \b1, [x1, #16]
	add  x1, x1, #64
	ldr  \a0, [x0], #16
	ldr  \a1, [x0], #16
	ldr  \a2, [x0], #16
	ldr  \a3, [x0], #16
	ldr  \a4, [x0], #16
	ldr  \a5, [x0], #16
	ldr  \a6, [x0], #16
	ldr  \a7, [x0], #16
	.endm

	.macro COMPUTE_CURR_PREFETCH_NEXT c_b0, c_b1, c_a0, c_a1, c_a2, c_a3, c_a4, c_a5, c_a6, c_a7, n_b0, n_b1, n_a0, n_a1, n_a2, n_a3, n_a4, n_a5, n_a6, n_a7
	ldr  \n_b0, [x1]
	ldr  \n_b1, [x1, #16]
	add  x1, x1, #64
	ldr  \n_a0, [x0], #16
	ldr  \n_a1, [x0], #16
	ldr  \n_a2, [x0], #16
	ldr  \n_a3, [x0], #16

	smmla v16.4s, \c_a0\().16b, \c_b0\().16b
	smmla v17.4s, \c_a0\().16b, \c_b1\().16b
	ldr  \n_a4, [x0], #16

	smmla v18.4s, \c_a1\().16b, \c_b0\().16b
	smmla v19.4s, \c_a1\().16b, \c_b1\().16b
	ldr  \n_a5, [x0], #16

	smmla v20.4s, \c_a2\().16b, \c_b0\().16b
	smmla v21.4s, \c_a2\().16b, \c_b1\().16b
	ldr  \n_a6, [x0], #16

	smmla v22.4s, \c_a3\().16b, \c_b0\().16b
	smmla v23.4s, \c_a3\().16b, \c_b1\().16b
	ldr  \n_a7, [x0], #16

	smmla v24.4s, \c_a4\().16b, \c_b0\().16b
	smmla v25.4s, \c_a4\().16b, \c_b1\().16b

	smmla v26.4s, \c_a5\().16b, \c_b0\().16b
	smmla v27.4s, \c_a5\().16b, \c_b1\().16b

	smmla v28.4s, \c_a6\().16b, \c_b0\().16b
	smmla v29.4s, \c_a6\().16b, \c_b1\().16b

	smmla v30.4s, \c_a7\().16b, \c_b0\().16b
	smmla v31.4s, \c_a7\().16b, \c_b1\().16b
	.endm

.Lm16n4_k32_loop:
	LOAD_SLICE q0, q1, q2, q3, q4, q5, q6, q7, q8, q9
	COMPUTE_CURR_PREFETCH_NEXT v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, q10, q11, q12, q13, q14, q15, q2, q3, q4, q5
	COMPUTE_CURR_PREFETCH_NEXT v10, v11, v12, v13, v14, v15, v2, v3, v4, v5, q0, q1, q6, q7, q8, q9, q12, q13, q14, q15
	COMPUTE_CURR_PREFETCH_NEXT v0, v1, v6, v7, v8, v9, v12, v13, v14, v15, q10, q11, q2, q3, q4, q5, q6, q7, q8, q9
	COMPUTE_M16N4 v2, v3, v4, v5, v6, v7, v8, v9, v10, v11
	subs x15, x15, #1
	b.ne .Lm16n4_k32_loop

	cbz  x14, .Lm16n4_store

.Lm16n4_k_tail_loop:
	LOAD_SLICE q0, q1, q2, q3, q4, q5, q6, q7, q8, q9
	COMPUTE_M16N4 v2, v3, v4, v5, v6, v7, v8, v9, v0, v1
	subs x14, x14, #1
	b.ne .Lm16n4_k_tail_loop

.Lm16n4_store:
	// Reset C_N_BASE.
	add  x11, x2, x5, lsl #2
	mov  x13, x6

	.macro STORE_C_PAIR acc0, acc1
	cmp  x13, #0
	b.le .Lm16n4_store_done
	zip1 v0.2d, \acc0\().2d, \acc1\().2d
	zip2 v1.2d, \acc0\().2d, \acc1\().2d
	str  q0, [x11]
	cmp  x13, #1
	b.eq .Lm16n4_store_done
	add  x12, x11, x10
	str  q1, [x12]
	add  x11, x11, x10, lsl #1
	sub  x13, x13, #2
	.endm

	STORE_C_PAIR v16, v17
	STORE_C_PAIR v18, v19
	STORE_C_PAIR v20, v21
	STORE_C_PAIR v22, v23
	STORE_C_PAIR v24, v25
	STORE_C_PAIR v26, v27
	STORE_C_PAIR v28, v29
	STORE_C_PAIR v30, v31

.Lm16n4_store_done:
	ldp d14, d15, [sp], #16
	ldp d12, d13, [sp], #16
	ldp d10, d11, [sp], #16
	ldp d8,  d9,  [sp], #16
	ret

#ifndef __APPLE__
	.size i8gemm_k_m16n4_packed_asm, .-i8gemm_k_m16n4_packed_asm
#endif
