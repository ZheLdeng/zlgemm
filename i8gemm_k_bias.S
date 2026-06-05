// i8gemm_k_bias.S — i8 GEMM kernel with fp32 bias addition (no C load)
// void i8gemm_k_ld_bias_f(const int8_t  *A,            // x0
//                         const int8_t  *B_reordered,  // x1
//                         float         *C,            // x2  fp32 output
//                         int8_t        *A_reorder,    // x3
//                         const gemm_params_t *params, // x4  {m,k,n,lda,ldb,ldc}
//                         const float   *bias);        // x5  bias[N] fp32
//
// Key design:
//   - Accumulators zero-initialised (no C read)
//   - smmla int8×int8 → int32 accumulate
//   - STORE: zip → scvtf (int32→fp32) → ldp bias → fadd → stp
//   - Bias is per-column, shared across rows; loaded once per N-block (8 values)
//   - Bias addition in fp32 domain avoids fp32→int32 precision loss

	.text
	.align 4

// ── Register aliases ──────────────────────────────────
#define A_ORIG      x0
#define B_ADDR      x1
#define C_ADDR_0    x2
#define M           x3
#define K           x4
#define N           x5
#define A_REORDER   x6
#define LDA         x7
#define LDC_B       x9
#define LDB         x8
#define BIAS_ADDR   x21    // bias pointer, callee-saved (saved/restored by INIT)

#define N_COUNT     x30
#define C_N_BASE    x27
#define M_COUNT     x10
#define B_ADDR_1    x19
#define B_N_BASE    x22
#define K_COUNT     x23
#define FIRST_N     x24
#define C_ROW_BASE  x25
#define C_M_BASE    x26
#define A_REORDER_M x28
#define A_ADDR      x29
#define A_ORIG_M    x20

// ── M_LOOP macros ─────────────────────────────────────
	.macro M_LOOP_BEGIN_1
	.endm
	.macro M_LOOP_BEGIN_2
	.endm
	.macro M_LOOP_BEGIN_4
	.endm
	.macro M_LOOP_BEGIN_8
	asr  M_COUNT, M, #3
	cmp  M_COUNT, #0
	b.eq 301f
400:
	.endm

	.macro M_LOOP_END_1
	.endm
	.macro M_LOOP_END_2
	.endm
	.macro M_LOOP_END_4
	.endm
	.macro M_LOOP_END_8
	add  A_ORIG_M, A_ORIG_M, LDA, lsl 3
	add  C_ROW_BASE, C_ROW_BASE, LDC_B, lsl 3
	subs M_COUNT, M_COUNT, #1
	b.gt 400b
301:
	.endm

// ── INIT / RESTORE ────────────────────────────────────
	.macro INIT
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

	.macro RESTORE
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

// ═══════════════════════════════════════════════════════
// LOAD_ZERO — zero-initialise accumulators (no C read)
// ═══════════════════════════════════════════════════════

// ── m=8: v9~v24 = 0 ──────────────────────────────────
	.macro LOAD_ZERO_8
	movi v9.4s, #0
	movi v10.4s, #0
	movi v11.4s, #0
	movi v12.4s, #0
	movi v13.4s, #0
	movi v14.4s, #0
	movi v15.4s, #0
	movi v16.4s, #0
	movi v17.4s, #0
	movi v18.4s, #0
	movi v19.4s, #0
	movi v20.4s, #0
	movi v21.4s, #0
	movi v22.4s, #0
	movi v23.4s, #0
	movi v24.4s, #0
	.endm

// ── m=1/2: v9~v12 = 0 ────────────────────────────────
	.macro LOAD_ZERO_2
	movi v9.4s, #0
	movi v10.4s, #0
	movi v11.4s, #0
	movi v12.4s, #0
	.endm

// ── m=1: v9~v12 = 0 (same as m=2, row1 is dummy) ─────
	.macro LOAD_ZERO_1
	movi v9.4s, #0
	movi v10.4s, #0
	movi v11.4s, #0
	movi v12.4s, #0
	.endm

// ── m=4: v9~v16 = 0 ──────────────────────────────────
	.macro LOAD_ZERO_4
	movi v9.4s, #0
	movi v10.4s, #0
	movi v11.4s, #0
	movi v12.4s, #0
	movi v13.4s, #0
	movi v14.4s, #0
	movi v15.4s, #0
	movi v16.4s, #0
	.endm

// ═══════════════════════════════════════════════════════
// STORE_C_*_BIAS_F — store with fp32 bias addition
// After zip+scvtf: load bias, fadd, then stp
// ═══════════════════════════════════════════════════════

// ── m=8: 8 rows × 8 cols, fp32 output + bias ─────────
	.macro STORE_C_ROWMAJOR_F_BIAS
	add  x11, C_M_BASE, LDC_B
	add  x12, C_M_BASE, LDC_B, lsl #1
	add  x14, C_M_BASE, LDC_B, lsl #2
	sub  x13, x14, LDC_B
	add  x15, x11, LDC_B, lsl #2
	add  x16, x12, LDC_B, lsl #2
	add  x17, x15, LDC_B, lsl #1

	// zip int32 accumulators → row-major layout
	zip1 v0.2d, v9.2d,  v10.2d
	zip1 v1.2d, v11.2d, v12.2d
	zip2 v2.2d, v9.2d,  v10.2d
	zip2 v3.2d, v11.2d, v12.2d

	zip1 v4.2d, v13.2d, v14.2d
	zip1 v5.2d, v15.2d, v16.2d
	zip2 v6.2d, v13.2d, v14.2d
	zip2 v7.2d, v15.2d, v16.2d

	zip1 v8.2d, v17.2d, v18.2d
	zip1 v25.2d, v19.2d, v20.2d
	zip2 v26.2d, v17.2d, v18.2d
	zip2 v27.2d, v19.2d, v20.2d

	zip1 v28.2d, v21.2d, v22.2d
	zip1 v29.2d, v23.2d, v24.2d
	zip2 v30.2d, v21.2d, v22.2d
	zip2 v31.2d, v23.2d, v24.2d

	// convert int32 → fp32
	scvtf v0.4s, v0.4s
	scvtf v1.4s, v1.4s
	scvtf v2.4s, v2.4s
	scvtf v3.4s, v3.4s
	scvtf v4.4s, v4.4s
	scvtf v5.4s, v5.4s
	scvtf v6.4s, v6.4s
	scvtf v7.4s, v7.4s
	scvtf v8.4s, v8.4s
	scvtf v25.4s, v25.4s
	scvtf v26.4s, v26.4s
	scvtf v27.4s, v27.4s
	scvtf v28.4s, v28.4s
	scvtf v29.4s, v29.4s
	scvtf v30.4s, v30.4s
	scvtf v31.4s, v31.4s

	// load bias: v16=[b0,b1,b2,b3], v17=[b4,b5,b6,b7]
	// (v16,v17 were accumulator regs, now dead after zip)
	ldp  q16, q17, [BIAS_ADDR]

	// fadd bias to each row (bias is per-column, shared across rows)
	fadd v0.4s, v0.4s, v16.4s     // row0 cols0-3 + bias[0:3]
	fadd v1.4s, v1.4s, v17.4s     // row0 cols4-7 + bias[4:7]
	fadd v2.4s, v2.4s, v16.4s     // row1 cols0-3
	fadd v3.4s, v3.4s, v17.4s     // row1 cols4-7
	fadd v4.4s, v4.4s, v16.4s     // row2 cols0-3
	fadd v5.4s, v5.4s, v17.4s     // row2 cols4-7
	fadd v6.4s, v6.4s, v16.4s     // row3 cols0-3
	fadd v7.4s, v7.4s, v17.4s     // row3 cols4-7
	fadd v8.4s, v8.4s, v16.4s     // row4 cols0-3
	fadd v25.4s, v25.4s, v17.4s   // row4 cols4-7
	fadd v26.4s, v26.4s, v16.4s   // row5 cols0-3
	fadd v27.4s, v27.4s, v17.4s   // row5 cols4-7
	fadd v28.4s, v28.4s, v16.4s   // row6 cols0-3
	fadd v29.4s, v29.4s, v17.4s   // row6 cols4-7
	fadd v30.4s, v30.4s, v16.4s   // row7 cols0-3
	fadd v31.4s, v31.4s, v17.4s   // row7 cols4-7

	stp  q0, q1, [C_M_BASE]
	stp  q2, q3, [x11]
	stp  q4, q5, [x12]
	stp  q6, q7, [x13]
	stp  q8, q25, [x14]
	stp  q26, q27, [x15]
	stp  q28, q29, [x16]
	stp  q30, q31, [x17]
	.endm

// ── m=1: 1 row × 8 cols, fp32 output + bias ──────────
	.macro STORE_C_1_F_BIAS
	zip1 v0.2d, v9.2d,  v10.2d      // row0 cols0-3
	zip1 v1.2d, v11.2d, v12.2d      // row0 cols4-7
	scvtf v0.4s, v0.4s
	scvtf v1.4s, v1.4s
	ldp  q2, q3, [BIAS_ADDR]        // bias (v2,v3 free after zip)
	fadd v0.4s, v0.4s, v2.4s        // cols0-3 + bias[0:3]
	fadd v1.4s, v1.4s, v3.4s        // cols4-7 + bias[4:7]
	stp  q0, q1, [C_M_BASE]
	.endm

// ── m=2: 2 rows × 8 cols, fp32 output + bias ─────────
	.macro STORE_C_2_F_BIAS
	zip1 v0.2d, v9.2d,  v10.2d      // row0 cols0-3
	zip1 v1.2d, v11.2d, v12.2d      // row0 cols4-7
	zip2 v2.2d, v9.2d,  v10.2d      // row1 cols0-3
	zip2 v3.2d, v11.2d, v12.2d      // row1 cols4-7
	scvtf v0.4s, v0.4s
	scvtf v1.4s, v1.4s
	scvtf v2.4s, v2.4s
	scvtf v3.4s, v3.4s
	ldp  q4, q5, [BIAS_ADDR]        // bias
	fadd v0.4s, v0.4s, v4.4s
	fadd v1.4s, v1.4s, v5.4s
	fadd v2.4s, v2.4s, v4.4s
	fadd v3.4s, v3.4s, v5.4s
	add  x11, C_M_BASE, LDC_B
	stp  q0, q1, [C_M_BASE]
	stp  q2, q3, [x11]
	.endm

// ── m=4: 4 rows × 8 cols, fp32 output + bias ─────────
	.macro STORE_C_4_F_BIAS
	add  x11, C_M_BASE, LDC_B
	add  x12, C_M_BASE, LDC_B, lsl #1
	add  x14, C_M_BASE, LDC_B, lsl #2
	sub  x13, x14, LDC_B

	zip1 v0.2d, v9.2d,  v10.2d
	zip1 v1.2d, v11.2d, v12.2d
	zip2 v2.2d, v9.2d,  v10.2d
	zip2 v3.2d, v11.2d, v12.2d

	zip1 v4.2d, v13.2d, v14.2d
	zip1 v5.2d, v15.2d, v16.2d
	zip2 v6.2d, v13.2d, v14.2d
	zip2 v7.2d, v15.2d, v16.2d

	scvtf v0.4s, v0.4s
	scvtf v1.4s, v1.4s
	scvtf v2.4s, v2.4s
	scvtf v3.4s, v3.4s
	scvtf v4.4s, v4.4s
	scvtf v5.4s, v5.4s
	scvtf v6.4s, v6.4s
	scvtf v7.4s, v7.4s

	// v16,v17 are dead after zip (used as accumulator sources above)
	// reuse v16,v17 for bias
	ldp  q16, q17, [BIAS_ADDR]
	fadd v0.4s, v0.4s, v16.4s
	fadd v1.4s, v1.4s, v17.4s
	fadd v2.4s, v2.4s, v16.4s
	fadd v3.4s, v3.4s, v17.4s
	fadd v4.4s, v4.4s, v16.4s
	fadd v5.4s, v5.4s, v17.4s
	fadd v6.4s, v6.4s, v16.4s
	fadd v7.4s, v7.4s, v17.4s

	stp  q0, q1, [C_M_BASE]
	stp  q2, q3, [x11]
	stp  q4, q5, [x12]
	stp  q6, q7, [x13]
	.endm

// ═══════════════════════════════════════════════════════
// CACHED PATH  按 mrows 特化
// (identical to i8gemm_k.S — smmla is the same)
// ═══════════════════════════════════════════════════════

// ── m=8 cached ───────────────────────────────────────
	.macro LOAD_A0_B0_8
	ldp q0, q1, [A_ADDR], #32
	ldp q2, q3, [A_ADDR], #32
	ldp q4, q5, [B_ADDR_1], #32
	ldp q6, q7, [B_ADDR_1], #32
	.endm

	.macro COMPUTE_A1_B1_8
	ldp q8,  q25, [A_ADDR], #32
	ldp q26, q27, [A_ADDR], #32
	ldp q28, q29, [B_ADDR_1], #32
	ldp q30, q31, [B_ADDR_1], #32

	smmla v9.4s,  v0.16b, v4.16b
	smmla v10.4s, v0.16b, v5.16b
	smmla v11.4s, v0.16b, v6.16b
	smmla v12.4s, v0.16b, v7.16b

	smmla v13.4s, v1.16b, v4.16b
	smmla v14.4s, v1.16b, v5.16b
	smmla v15.4s, v1.16b, v6.16b
	smmla v16.4s, v1.16b, v7.16b

	smmla v17.4s, v2.16b, v4.16b
	smmla v18.4s, v2.16b, v5.16b
	smmla v19.4s, v2.16b, v6.16b
	smmla v20.4s, v2.16b, v7.16b

	smmla v21.4s, v3.16b, v4.16b
	smmla v22.4s, v3.16b, v5.16b
	smmla v23.4s, v3.16b, v6.16b
	smmla v24.4s, v3.16b, v7.16b
	.endm

	.macro COMPUTE_A2_B2_8
	ldp q0, q1, [A_ADDR], #32
	ldp q2, q3, [A_ADDR], #32
	ldp q4, q5, [B_ADDR_1], #32
	ldp q6, q7, [B_ADDR_1], #32

	smmla v9.4s,  v8.16b, v28.16b
	smmla v10.4s, v8.16b, v29.16b
	smmla v11.4s, v8.16b, v30.16b
	smmla v12.4s, v8.16b, v31.16b

	smmla v13.4s, v25.16b, v28.16b
	smmla v14.4s, v25.16b, v29.16b
	smmla v15.4s, v25.16b, v30.16b
	smmla v16.4s, v25.16b, v31.16b

	smmla v17.4s, v26.16b, v28.16b
	smmla v18.4s, v26.16b, v29.16b
	smmla v19.4s, v26.16b, v30.16b
	smmla v20.4s, v26.16b, v31.16b

	smmla v21.4s, v27.16b, v28.16b
	smmla v22.4s, v27.16b, v29.16b
	smmla v23.4s, v27.16b, v30.16b
	smmla v24.4s, v27.16b, v31.16b
	.endm

	.macro COMPUTE_A3_B3_8
	smmla v9.4s,  v8.16b, v28.16b
	smmla v10.4s, v8.16b, v29.16b
	smmla v11.4s, v8.16b, v30.16b
	smmla v12.4s, v8.16b, v31.16b

	smmla v13.4s, v25.16b, v28.16b
	smmla v14.4s, v25.16b, v29.16b
	smmla v15.4s, v25.16b, v30.16b
	smmla v16.4s, v25.16b, v31.16b

	smmla v17.4s, v26.16b, v28.16b
	smmla v18.4s, v26.16b, v29.16b
	smmla v19.4s, v26.16b, v30.16b
	smmla v20.4s, v26.16b, v31.16b

	smmla v21.4s, v27.16b, v28.16b
	smmla v22.4s, v27.16b, v29.16b
	smmla v23.4s, v27.16b, v30.16b
	smmla v24.4s, v27.16b, v31.16b
	.endm

// ── m=1/2 cached ──────────────────────────────────────
	.macro LOAD_A0_B0_2
	ldr  q0, [A_ADDR], #16
	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32
	.endm

	.macro COMPUTE_A1_B1_2
	ldr  q8, [A_ADDR], #16
	ldp  q28, q29, [B_ADDR_1], #32
	ldp  q30, q31, [B_ADDR_1], #32

	smmla v9.4s,  v0.16b, v4.16b
	smmla v10.4s, v0.16b, v5.16b
	smmla v11.4s, v0.16b, v6.16b
	smmla v12.4s, v0.16b, v7.16b
	.endm

	.macro COMPUTE_A2_B2_2
	ldr  q0, [A_ADDR], #16
	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32

	smmla v9.4s,  v8.16b, v28.16b
	smmla v10.4s, v8.16b, v29.16b
	smmla v11.4s, v8.16b, v30.16b
	smmla v12.4s, v8.16b, v31.16b
	.endm

	.macro COMPUTE_A3_B3_2
	smmla v9.4s,  v8.16b, v28.16b
	smmla v10.4s, v8.16b, v29.16b
	smmla v11.4s, v8.16b, v30.16b
	smmla v12.4s, v8.16b, v31.16b
	.endm

// ── m=4 cached ────────────────────────────────────────
	.macro LOAD_A0_B0_4
	ldp  q0, q1, [A_ADDR], #32
	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32
	.endm

	.macro COMPUTE_A1_B1_4
	ldp  q8, q25, [A_ADDR], #32
	ldp  q28, q29, [B_ADDR_1], #32
	ldp  q30, q31, [B_ADDR_1], #32

	smmla v9.4s,  v0.16b, v4.16b
	smmla v10.4s, v0.16b, v5.16b
	smmla v11.4s, v0.16b, v6.16b
	smmla v12.4s, v0.16b, v7.16b

	smmla v13.4s, v1.16b, v4.16b
	smmla v14.4s, v1.16b, v5.16b
	smmla v15.4s, v1.16b, v6.16b
	smmla v16.4s, v1.16b, v7.16b
	.endm

	.macro COMPUTE_A2_B2_4
	ldp  q0, q1, [A_ADDR], #32
	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32

	smmla v9.4s,  v8.16b,  v28.16b
	smmla v10.4s, v8.16b,  v29.16b
	smmla v11.4s, v8.16b,  v30.16b
	smmla v12.4s, v8.16b,  v31.16b

	smmla v13.4s, v25.16b, v28.16b
	smmla v14.4s, v25.16b, v29.16b
	smmla v15.4s, v25.16b, v30.16b
	smmla v16.4s, v25.16b, v31.16b
	.endm

	.macro COMPUTE_A3_B3_4
	smmla v9.4s,  v8.16b,  v28.16b
	smmla v10.4s, v8.16b,  v29.16b
	smmla v11.4s, v8.16b,  v30.16b
	smmla v12.4s, v8.16b,  v31.16b

	smmla v13.4s, v25.16b, v28.16b
	smmla v14.4s, v25.16b, v29.16b
	smmla v15.4s, v25.16b, v30.16b
	smmla v16.4s, v25.16b, v31.16b
	.endm

// ═══════════════════════════════════════════════════════
// STRIDED PATH  按 mrows 特化
// (identical to i8gemm_k.S — same A reordering logic)
// ═══════════════════════════════════════════════════════

// ── m=8 strided ───────────────────────────────────────
	.macro LOAD_A0_B0_STRIDED_8
	add  x11, A_ADDR, LDA
	add  x12, A_ADDR, LDA, lsl #1
	add  x14, A_ADDR, LDA, lsl #2
	sub  x13, x14, LDA
	add  x15, x11, LDA, lsl #2
	add  x16, x12, LDA, lsl #2
	add  x17, x15, LDA, lsl #1

	ld1  {v0.d}[0], [A_ADDR]
	ld1  {v0.d}[1], [x11]
	ld1  {v1.d}[0], [x12]
	ld1  {v1.d}[1], [x13]
	ld1  {v2.d}[0], [x14]
	ld1  {v2.d}[1], [x15]
	ld1  {v3.d}[0], [x16]
	ld1  {v3.d}[1], [x17]

	stp  q0, q1, [A_REORDER], #32
	stp  q2, q3, [A_REORDER], #32

	add  A_ADDR, A_ADDR, #8

	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32
	.endm

	.macro COMPUTE_A1_B1_STRIDED_8
	add  x11, A_ADDR, LDA
	add  x12, A_ADDR, LDA, lsl #1
	add  x14, A_ADDR, LDA, lsl #2
	sub  x13, x14, LDA
	add  x15, x11, LDA, lsl #2
	add  x16, x12, LDA, lsl #2
	add  x17, x15, LDA, lsl #1

	ld1  {v8.d}[0],  [A_ADDR]
	ld1  {v8.d}[1],  [x11]
	ld1  {v25.d}[0], [x12]
	ld1  {v25.d}[1], [x13]
	ld1  {v26.d}[0], [x14]
	ld1  {v26.d}[1], [x15]
	ld1  {v27.d}[0], [x16]
	ld1  {v27.d}[1], [x17]

	stp  q8,  q25, [A_REORDER], #32
	stp  q26, q27, [A_REORDER], #32

	add  A_ADDR, A_ADDR, #8

	ldp  q28, q29, [B_ADDR_1], #32
	ldp  q30, q31, [B_ADDR_1], #32

	smmla v9.4s,  v0.16b, v4.16b
	smmla v10.4s, v0.16b, v5.16b
	smmla v11.4s, v0.16b, v6.16b
	smmla v12.4s, v0.16b, v7.16b

	smmla v13.4s, v1.16b, v4.16b
	smmla v14.4s, v1.16b, v5.16b
	smmla v15.4s, v1.16b, v6.16b
	smmla v16.4s, v1.16b, v7.16b

	smmla v17.4s, v2.16b, v4.16b
	smmla v18.4s, v2.16b, v5.16b
	smmla v19.4s, v2.16b, v6.16b
	smmla v20.4s, v2.16b, v7.16b

	smmla v21.4s, v3.16b, v4.16b
	smmla v22.4s, v3.16b, v5.16b
	smmla v23.4s, v3.16b, v6.16b
	smmla v24.4s, v3.16b, v7.16b
	.endm

	.macro COMPUTE_A2_B2_STRIDED_8
	add  x11, A_ADDR, LDA
	add  x12, A_ADDR, LDA, lsl #1
	add  x14, A_ADDR, LDA, lsl #2
	sub  x13, x14, LDA
	add  x15, x11, LDA, lsl #2
	add  x16, x12, LDA, lsl #2
	add  x17, x15, LDA, lsl #1

	ld1  {v0.d}[0], [A_ADDR]
	ld1  {v0.d}[1], [x11]
	ld1  {v1.d}[0], [x12]
	ld1  {v1.d}[1], [x13]
	ld1  {v2.d}[0], [x14]
	ld1  {v2.d}[1], [x15]
	ld1  {v3.d}[0], [x16]
	ld1  {v3.d}[1], [x17]

	stp  q0, q1, [A_REORDER], #32
	stp  q2, q3, [A_REORDER], #32

	add  A_ADDR, A_ADDR, #8

	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32

	smmla v9.4s,  v8.16b, v28.16b
	smmla v10.4s, v8.16b, v29.16b
	smmla v11.4s, v8.16b, v30.16b
	smmla v12.4s, v8.16b, v31.16b

	smmla v13.4s, v25.16b, v28.16b
	smmla v14.4s, v25.16b, v29.16b
	smmla v15.4s, v25.16b, v30.16b
	smmla v16.4s, v25.16b, v31.16b

	smmla v17.4s, v26.16b, v28.16b
	smmla v18.4s, v26.16b, v29.16b
	smmla v19.4s, v26.16b, v30.16b
	smmla v20.4s, v26.16b, v31.16b

	smmla v21.4s, v27.16b, v28.16b
	smmla v22.4s, v27.16b, v29.16b
	smmla v23.4s, v27.16b, v30.16b
	smmla v24.4s, v27.16b, v31.16b
	.endm

// ── m=1 strided ───────────────────────────────────────
	.macro LOAD_A0_B0_STRIDED_1
	ldr  d0, [A_ADDR]
	str  q0, [A_REORDER], #16
	add  A_ADDR, A_ADDR, #8

	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32
	.endm

	.macro COMPUTE_A1_B1_STRIDED_1
	ldr  d8, [A_ADDR]
	str  q8, [A_REORDER], #16
	add  A_ADDR, A_ADDR, #8

	ldp  q28, q29, [B_ADDR_1], #32
	ldp  q30, q31, [B_ADDR_1], #32

	smmla v9.4s,  v0.16b, v4.16b
	smmla v10.4s, v0.16b, v5.16b
	smmla v11.4s, v0.16b, v6.16b
	smmla v12.4s, v0.16b, v7.16b
	.endm

	.macro COMPUTE_A2_B2_STRIDED_1
	ldr  d0, [A_ADDR]
	mov v0.d[1], xzr
	str  q0, [A_REORDER], #16
	add  A_ADDR, A_ADDR, #8

	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32

	smmla v9.4s,  v8.16b, v28.16b
	smmla v10.4s, v8.16b, v29.16b
	smmla v11.4s, v8.16b, v30.16b
	smmla v12.4s, v8.16b, v31.16b
	.endm

// ── m=2 strided ───────────────────────────────────────
	.macro LOAD_A0_B0_STRIDED_2
	add  x11, A_ADDR, LDA
	ldr  d0, [A_ADDR]
	ld1  {v0.d}[1], [x11]
	str  q0, [A_REORDER], #16
	add  A_ADDR, A_ADDR, #8

	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32
	.endm

	.macro COMPUTE_A1_B1_STRIDED_2
	add  x11, A_ADDR, LDA
	ldr  d8, [A_ADDR]
	ld1  {v8.d}[1], [x11]
	str  q8, [A_REORDER], #16
	add  A_ADDR, A_ADDR, #8

	ldp  q28, q29, [B_ADDR_1], #32
	ldp  q30, q31, [B_ADDR_1], #32

	smmla v9.4s,  v0.16b, v4.16b
	smmla v10.4s, v0.16b, v5.16b
	smmla v11.4s, v0.16b, v6.16b
	smmla v12.4s, v0.16b, v7.16b
	.endm

	.macro COMPUTE_A2_B2_STRIDED_2
	add  x11, A_ADDR, LDA
	ldr  d0, [A_ADDR]
	ld1  {v0.d}[1], [x11]
	str  q0, [A_REORDER], #16
	add  A_ADDR, A_ADDR, #8

	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32

	smmla v9.4s,  v8.16b, v28.16b
	smmla v10.4s, v8.16b, v29.16b
	smmla v11.4s, v8.16b, v30.16b
	smmla v12.4s, v8.16b, v31.16b
	.endm

// ── m=4 strided ───────────────────────────────────────
	.macro LOAD_A0_B0_STRIDED_4
	add  x11, A_ADDR, LDA
	add  x12, A_ADDR, LDA, lsl #1
	add  x14, A_ADDR, LDA, lsl #2
	sub  x13, x14, LDA

	ldr  d0, [A_ADDR]
	ld1  {v0.d}[1], [x11]
	ldr  d1, [x12]
	ld1  {v1.d}[1], [x13]

	stp  q0, q1, [A_REORDER], #32
	add  A_ADDR, A_ADDR, #8

	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32
	.endm

	.macro COMPUTE_A1_B1_STRIDED_4
	add  x11, A_ADDR, LDA
	add  x12, A_ADDR, LDA, lsl #1
	add  x14, A_ADDR, LDA, lsl #2
	sub  x13, x14, LDA

	ldr  d8, [A_ADDR]
	ld1  {v8.d}[1], [x11]
	ldr  d25, [x12]
	ld1  {v25.d}[1], [x13]

	stp  q8, q25, [A_REORDER], #32
	add  A_ADDR, A_ADDR, #8

	ldp  q28, q29, [B_ADDR_1], #32
	ldp  q30, q31, [B_ADDR_1], #32

	smmla v9.4s,  v0.16b, v4.16b
	smmla v10.4s, v0.16b, v5.16b
	smmla v11.4s, v0.16b, v6.16b
	smmla v12.4s, v0.16b, v7.16b

	smmla v13.4s, v1.16b, v4.16b
	smmla v14.4s, v1.16b, v5.16b
	smmla v15.4s, v1.16b, v6.16b
	smmla v16.4s, v1.16b, v7.16b
	.endm

	.macro COMPUTE_A2_B2_STRIDED_4
	add  x11, A_ADDR, LDA
	add  x12, A_ADDR, LDA, lsl #1
	add  x14, A_ADDR, LDA, lsl #2
	sub  x13, x14, LDA

	ldr  d0, [A_ADDR]
	ld1  {v0.d}[1], [x11]
	ldr  d1, [x12]
	ld1  {v1.d}[1], [x13]

	stp  q0, q1, [A_REORDER], #32
	add  A_ADDR, A_ADDR, #8

	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32

	smmla v9.4s,  v8.16b,  v28.16b
	smmla v10.4s, v8.16b,  v29.16b
	smmla v11.4s, v8.16b,  v30.16b
	smmla v12.4s, v8.16b,  v31.16b

	smmla v13.4s, v25.16b, v28.16b
	smmla v14.4s, v25.16b, v29.16b
	smmla v15.4s, v25.16b, v30.16b
	smmla v16.4s, v25.16b, v31.16b
	.endm

// ═══════════════════════════════════════════════════════
// GEMM_BODY_BIAS — unified function body with bias support
//
// Differences from GEMM_BODY:
//   1. Loads BIAS_ADDR from stack at offset 176 after INIT
//   2. Resets BIAS_ADDR for each M block (reload from stack)
//   3. Advances BIAS_ADDR by 32 per N block in epilogue
//   4. Uses LOAD_ZERO_* for accumulator init (no C read)
//
// Parameters:
//   \func_name  : symbol name
//   \mrows      : 1 / 2 / 4 / 8
//   \cmpsfx     : 2 / 4 / 8  — cached COMPUTE/LOAD macro suffix
//   \load_z     : LOAD_ZERO macro name
//   \store_c    : STORE_C_*_BIAS_F macro name
//   \ldc_shift  : ldc left-shift (2 = *4 bytes)
//   \c_adv      : C column advance in bytes (32 = 8×f32)
// ═══════════════════════════════════════════════════════

	.macro GEMM_BODY_BIAS func_name, mrows, cmpsfx, load_z, store_c, ldc_shift, c_adv

#ifdef __APPLE__
	.global _\func_name
_\func_name:
#else
	.global \func_name
\func_name:
#endif
	INIT

	// New calling convention: x3 = A_REORDER, x4 = PARAMS, x5 = bias
	mov  A_REORDER, x3              // save A_REORDER → x6
	str  x5, [sp, #-16]!            // push original bias (for M-loop reload)
	mov  BIAS_ADDR, x5              // current bias pointer → x21

	// Load m,k,n,lda,ldb,ldc from params struct (6×int32 = 24 bytes)
	mov  x11, x4                    // keep params pointer before K overwrites x4
	ldp  w3, w4, [x11]              // M, K
	ldp  w5, w7, [x11, #8]          // N, LDA (byte stride, no shift needed)
	ldp  w8, w12, [x11, #16]        // LDB, LDC

	lsl  LDC_B, x12, #\ldc_shift

	mov  A_ORIG_M, A_ORIG
	mov  C_ROW_BASE, C_ADDR_0

	M_LOOP_BEGIN_\mrows

	// Reset per-M-block state; reload bias pointer for this M block
	ldr  BIAS_ADDR, [sp]            // reload original bias from our slot
	mov  A_REORDER_M, A_REORDER
	mov  FIRST_N, #1
	mov  C_N_BASE, C_ROW_BASE
	mov  B_N_BASE, B_ADDR

.L\func_name\()_n_loop:
	asr  N_COUNT, N, #3
	cmp  N_COUNT, #0
	b.eq .L\func_name\()_end

.L\func_name\()_n_iter:
	mov  B_ADDR_1, B_N_BASE
	mov  C_M_BASE, C_N_BASE

	cmp  FIRST_N, #1
	b.eq .L\func_name\()_strided

// ── CACHED PATH ──────────────────────────────────────
	mov  A_ADDR, A_REORDER_M

	\load_z

	asr  K_COUNT, K, #4
	cmp  K_COUNT, #0
	b.eq .L\func_name\()_end
	subs K_COUNT, K_COUNT, #1

	LOAD_A0_B0_\cmpsfx

	b.eq .L\func_name\()_cached_tail

.L\func_name\()_cached_loop:
	COMPUTE_A1_B1_\cmpsfx
	COMPUTE_A2_B2_\cmpsfx
	subs K_COUNT, K_COUNT, #1
	b.gt .L\func_name\()_cached_loop

.L\func_name\()_cached_tail:
	COMPUTE_A1_B1_\cmpsfx
	COMPUTE_A3_B3_\cmpsfx

	b .L\func_name\()_epilogue

// ── STRIDED PATH ─────────────────────────────────────
.L\func_name\()_strided:
	mov  A_ADDR, A_ORIG_M

	\load_z

	asr  K_COUNT, K, #4
	cmp  K_COUNT, #0
	b.eq .L\func_name\()_end
	subs K_COUNT, K_COUNT, #1

	LOAD_A0_B0_STRIDED_\mrows

	b.eq .L\func_name\()_strided_tail

.L\func_name\()_strided_loop:
	COMPUTE_A1_B1_STRIDED_\mrows
	COMPUTE_A2_B2_STRIDED_\mrows
	subs K_COUNT, K_COUNT, #1
	b.gt .L\func_name\()_strided_loop

.L\func_name\()_strided_tail:
	COMPUTE_A1_B1_STRIDED_\mrows
	COMPUTE_A3_B3_\cmpsfx

	mov  FIRST_N, #0

// ── EPILOGUE ─────────────────────────────────────────
.L\func_name\()_epilogue:
	\store_c
	add  C_N_BASE, C_N_BASE, #\c_adv
	add  B_N_BASE, B_N_BASE, LDB, lsl #3
	add  BIAS_ADDR, BIAS_ADDR, #32     // advance bias by 8×f32

	subs N_COUNT, N_COUNT, #1
	b.gt .L\func_name\()_n_iter

	M_LOOP_END_\mrows

.L\func_name\()_end:
	add  sp, sp, #16                // pop saved bias
	RESTORE
	ret

	.endm   // GEMM_BODY_BIAS

// ═══════════════════════════════════════════════════════
// Four function instantiations
// ═══════════════════════════════════════════════════════

// Main kernel (m=8)
GEMM_BODY_BIAS i8gemm_k_ld_bias_f,  8, 8, LOAD_ZERO_8, STORE_C_ROWMAJOR_F_BIAS, 2, 32

// Tail kernels (m=1,2,4)
GEMM_BODY_BIAS i8gemm_k_ld1_bias_f, 1, 2, LOAD_ZERO_2, STORE_C_1_F_BIAS,        2, 32
GEMM_BODY_BIAS i8gemm_k_ld2_bias_f, 2, 2, LOAD_ZERO_2, STORE_C_2_F_BIAS,        2, 32
GEMM_BODY_BIAS i8gemm_k_ld4_bias_f, 4, 4, LOAD_ZERO_4, STORE_C_4_F_BIAS,        2, 32
