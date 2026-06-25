// i8gemm_msplit_k.S -- microkernel for the M-split / split-K lib.
//
// Identical compute to i8gemm_k_hybrid (8-row x 2VL tile, double-buffered smmla,
// contiguous deinterleave store, A read directly from row-major layout), with ONE
// change: the byte stride between adjacent N-panels of B_reo is taken from the
// params.ldb field instead of being computed as K*n_tile. This lets a caller sweep
// a *K-slice* (params.k = slice length) across the *full N* in a single call while
// B_reo is still laid out for the full K (n-panel stride = K_full * n_tile). That is
// what the split-K small-M path needs; for a plain M-split the caller passes
// ldb = K*n_tile and k = K, recovering the original hybrid behaviour exactly.
//
// void i8gemm_msk(const int8_t *A, const int8_t *B_reo, int32_t *C,
//                 int8_t *unused, const gemm_params_t *p);
//   params: m, k, n, lda, ldb, ldc.
//     m   = rows in this M-band         (processed in 8-row blocks)
//     k   = K-slice length              (multiple of 8)
//     n   = number of columns           (full N for this band)
//     lda = bytes per A row             (= K_full, so A+k0 selects the slice)
//     ldb = bytes between N-panels of B (= K_full * n_tile)
//     ldc = bytes-per-row/4 of C        (row stride in int32 elements)
//   B_reo pointer must already point at the K-slice (B_reo + (k0/8)*8*n_tile).

	.text
	.arch armv8.6-a+sve+i8mm

	.macro MSAVE
	stp x19, x20, [sp, #-16]!
	stp x21, x22, [sp, #-16]!
	stp d8,  d9,  [sp, #-16]!
	stp d10, d11, [sp, #-16]!
	stp d12, d13, [sp, #-16]!
	stp d14, d15, [sp, #-16]!
	.endm
	.macro MREST
	ldp d14, d15, [sp], #16
	ldp d12, d13, [sp], #16
	ldp d10, d11, [sp], #16
	ldp d8,  d9,  [sp], #16
	ldp x21, x22, [sp], #16
	ldp x19, x20, [sp], #16
	.endm

	.macro MLOAD_A zreg, rp
	cmp x3, #(\rp * 2 + 1)
	b.gt .Lmla_\rp\()_both
	cmp x3, #(\rp * 2)
	b.le .Lmla_\rp\()_zero
	mov x22, #(\rp * 2)
	madd x22, x22, x11, x21
	ldr d0, [x22]
	mov v0.d[1], xzr
	b .Lmla_\rp\()_dup
.Lmla_\rp\()_both:
	mov x22, #(\rp * 2)
	madd x22, x22, x11, x21
	ldr d0, [x22]
	add x22, x22, x11
	ld1 {v0.d}[1], [x22]
.Lmla_\rp\()_dup:
	dup \zreg\().q, z0.q[0]
	b .Lmla_\rp\()_done
.Lmla_\rp\()_zero:
	mov \zreg\().b, #0
.Lmla_\rp\()_done:
	.endm

	.macro MMMLA_ROW areg, c0, c1, c2, c3
	smmla \c0\().s, \areg\().b, z4.b
	smmla \c1\().s, \areg\().b, z5.b
	smmla \c2\().s, \areg\().b, z6.b
	smmla \c3\().s, \areg\().b, z7.b
	.endm

	.macro MSTORE_PAIR rp, a0, a1, a2, a3
	cmp x3, #(\rp * 2)
	b.le .Lms_\rp\()_done
	trn1 z0.d, \a0\().d, \a1\().d
	trn1 z1.d, \a2\().d, \a3\().d
	mov x4, #(\rp * 2)
	mul x7, x4, x12
	add x7, x18, x7, lsl #2
	st1w z0.s, p2, [x7]
	add x7, x7, #16
	st1w z1.s, p2, [x7]
	add x7, x7, #16
	mov z2.d, z0.d
	ext z2.b, z2.b, z2.b, #16
	st1w z2.s, p2, [x7]
	add x7, x7, #16
	mov z3.d, z1.d
	ext z3.b, z3.b, z3.b, #16
	st1w z3.s, p2, [x7]
	cmp x3, #(\rp * 2 + 1)
	b.le .Lms_\rp\()_done
	trn2 z0.d, \a0\().d, \a1\().d
	trn2 z1.d, \a2\().d, \a3\().d
	mov x4, #(\rp * 2 + 1)
	mul x7, x4, x12
	add x7, x18, x7, lsl #2
	st1w z0.s, p2, [x7]
	add x7, x7, #16
	st1w z1.s, p2, [x7]
	add x7, x7, #16
	mov z2.d, z0.d
	ext z2.b, z2.b, z2.b, #16
	st1w z2.s, p2, [x7]
	add x7, x7, #16
	mov z3.d, z1.d
	ext z3.b, z3.b, z3.b, #16
	st1w z3.s, p2, [x7]
.Lms_\rp\()_done:
	.endm

	.global i8gemm_msk
	.type i8gemm_msk, %function
i8gemm_msk:
	MSAVE
	ldr w8,  [x4, #0]         // M (this band)
	ldr w9,  [x4, #4]         // K (slice length)
	ldr w10, [x4, #8]         // N
	ldr w11, [x4, #12]        // lda (bytes per A row = K_full)
	ldr w14, [x4, #16]        // ldb (bytes between N-panels of B_reo)
	ldr w12, [x4, #20]        // ldc (C row stride in int32 elements)
	mov x19, x1               // B_reo base (already at K-slice offset)

	ptrue p0.b
	ptrue p2.s, vl4

	cntb x13
	lsr x13, x13, #1          // n_tile = VL/2 (cols)

	mov x15, #0               // m0
.Lmsk_m:
	cmp x15, x8
	b.ge .Lmsk_done
	sub x3, x8, x15
	cmp x3, #8
	mov x20, #8
	csel x3, x3, x20, le      // m_now = min(M-m0, 8)

	mov x16, #0               // n0
	mov x17, x19              // current B N-panel
.Lmsk_n:
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

	mov x6, x17               // B K pointer for this panel
	mul x21, x15, x11
	add x21, x0, x21          // A + m0*lda  (k0 already folded into x0 by caller)
	mov x4, #0                // kb
.Lmsk_k:
	cmp x4, x9
	b.ge .Lmsk_store
	ld1b z4.b, p0/z, [x6]
	ld1b z5.b, p0/z, [x6, #1, mul vl]
	ld1b z6.b, p0/z, [x6, #2, mul vl]
	ld1b z7.b, p0/z, [x6, #3, mul vl]
	prfm pldl1keep, [x6, #512]
	MLOAD_A z0, 0
	MMMLA_ROW z0, z16, z17, z18, z19
	MLOAD_A z1, 1
	MMMLA_ROW z1, z20, z21, z22, z23
	MLOAD_A z2, 2
	MMMLA_ROW z2, z24, z25, z26, z27
	MLOAD_A z3, 3
	MMMLA_ROW z3, z28, z29, z30, z31
	add x4, x4, #8
	add x21, x21, #8
	addvl x6, x6, #4
	b .Lmsk_k

.Lmsk_store:
	madd x18, x15, x12, x16
	add x18, x2, x18, lsl #2
	MSTORE_PAIR 0, z16, z17, z18, z19
	MSTORE_PAIR 1, z20, z21, z22, z23
	MSTORE_PAIR 2, z24, z25, z26, z27
	MSTORE_PAIR 3, z28, z29, z30, z31

	add x16, x16, x13
	add x17, x17, x14         // next N-panel: ldb bytes ahead
	cmp x16, x10
	b.lt .Lmsk_n

	add x15, x15, #8
	b .Lmsk_m
.Lmsk_done:
	MREST
	ret
	.size i8gemm_msk, .-i8gemm_msk

	.section .note.GNU-stack,"",@progbits
