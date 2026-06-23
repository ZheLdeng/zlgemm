// i8gemm_k.S — unified i8 GEMM kernel (main m=8 + tail m=1,2,4)
// void i8gemm_k_ld(const int8_t  *A,            // x0  原始 A, M×K 行主序（大矩阵的子区域）
//                  const int8_t  *B_reordered,  // x1  已 pack 的 B（大 packed B 的子区域）
//                  int32_t       *C,            // x2  行主序输出（大矩阵的子区域）
//                  int8_t        *A_reorder,    // x3  A 重排输出 buffer
//                  const gemm_params_t *params);// x4  {m,k,n,lda,ldb,ldc} 结构体指针
//
// 循环顺序: M-N-K (最外层M, 中间N, 最内层K)
// A 重排策略:
//   每个 M block 的第一个 N iteration 从原始 A 跨 stride (lda) 读取，
//   写入 A_reorder; 后续 N iteration 直接从 A_reorder 连续读取。
// B: 预先 pack 好（外部调用者负责），N-block stride = ldb * 8 bytes。
//
// 内核每次处理 8×8 输出块，v9~v24 的 smmla 格式：
//   v9  = [C[r0][c0], C[r0][c1], C[r1][c0], C[r1][c1]]
//   v10 = [C[r0][c2], C[r0][c3], C[r1][c2], C[r1][c3]]
//   v11 = [C[r0][c4], C[r0][c5], C[r1][c4], C[r1][c5]]
//   v12 = [C[r0][c6], C[r0][c7], C[r1][c6], C[r1][c7]]
//   v13~v16: r2/r3 同上, v17~v20: r4/r5, v21~v24: r6/r7

	.text
	.align 4

// ── 寄存器别名（单一定义，全局复用）──────────────────────────────────
#define A_ORIG      x0
#define B_ADDR      x1
#define C_ADDR_0    x2
#define M           x3
#define K           x4
#define N           x5
#define A_REORDER   x6
#define LDA         x7    // A row stride in bytes (int8 = 1 byte/element)
#define LDC_B       x9    // C row stride in bytes = ldc * 4
#define LDB         x8    // K_big of packed B (loaded from stack)

#define N_COUNT     x30
#define C_N_BASE    x27
#define M_COUNT     x10
#define B_ADDR_1    x19
#define B_N_BASE    x22
#define K_COUNT     x23
#define FIRST_N     x24   // flag: 1 = 当前 M block 的首个 N iteration
#define C_ROW_BASE  x25
#define C_M_BASE    x26
#define A_REORDER_M x28   // 当前 M block 在 A_reorder 中的基地址
#define A_ADDR      x29   // A 加载指针（指向 orig A 或 reorder buffer）
#define A_ORIG_M    x20   // 当前 M block 在原始 A 中的基地址

// ── M_LOOP 宏：m=8 用真循环，m=1/2/4 用空宏 ──────────────────────
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

	.macro ADVANCE_PREPACKED_A_0
	add  A_REORDER, A_REORDER, K, lsl #3
	.endm
	.macro ADVANCE_PREPACKED_A_1
	.endm

// ── INIT / RESTORE ────────────────────────────────────────────
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

// ════════════════════════════════════════════════════════════════════
// LOAD_C / STORE_C  按行数特化
// ════════════════════════════════════════════════════════════════════

// ── zero-C fast path: public int32 dispatch documents C as zero-init.
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

	.macro LOAD_ZERO_2
	movi v9.4s, #0
	movi v10.4s, #0
	movi v11.4s, #0
	movi v12.4s, #0
	.endm

	.macro LOAD_ZERO_1
	movi v9.4s, #0
	movi v10.4s, #0
	movi v11.4s, #0
	movi v12.4s, #0
	.endm

// ── m=8：从行主序 C 加载 8×8 int32 → blocked v9~v24 ──────────────
	.macro LOAD_C_ROWMAJOR
	add  x11, C_M_BASE, LDC_B             // 1*LDC
	add  x12, C_M_BASE, LDC_B, lsl #1     // 2*LDC
	add  x14, C_M_BASE, LDC_B, lsl #2     // 4*LDC
	sub  x13, x14, LDC_B                  // 3*LDC = 4-1
	add  x15, x11, LDC_B, lsl #2          // 5*LDC = 1+4
	add  x16, x12, LDC_B, lsl #2          // 6*LDC = 2+4
	add  x17, x15, LDC_B, lsl #1          // 7*LDC = 5+2

	ldp  q0, q1, [C_M_BASE]
	ldp  q2, q3, [x11]
	ldp  q4, q5, [x12]
	ldp  q6, q7, [x13]
	ldp  q8, q25, [x14]
	ldp  q26, q27, [x15]
	ldp  q28, q29, [x16]
	ldp  q30, q31, [x17]

	zip1 v9.2d,  v0.2d, v2.2d
	zip2 v10.2d, v0.2d, v2.2d
	zip1 v11.2d, v1.2d, v3.2d
	zip2 v12.2d, v1.2d, v3.2d

	zip1 v13.2d, v4.2d, v6.2d
	zip2 v14.2d, v4.2d, v6.2d
	zip1 v15.2d, v5.2d, v7.2d
	zip2 v16.2d, v5.2d, v7.2d

	zip1 v17.2d, v8.2d,  v26.2d
	zip2 v18.2d, v8.2d,  v26.2d
	zip1 v19.2d, v25.2d, v27.2d
	zip2 v20.2d, v25.2d, v27.2d

	zip1 v21.2d, v28.2d, v30.2d
	zip2 v22.2d, v28.2d, v30.2d
	zip1 v23.2d, v29.2d, v31.2d
	zip2 v24.2d, v29.2d, v31.2d
	.endm

// ── m=8：将 blocked v9~v24 按行主序写回 C（int32）────────────────
	.macro STORE_C_ROWMAJOR
	add  x11, C_M_BASE, LDC_B
	add  x12, C_M_BASE, LDC_B, lsl #1
	add  x14, C_M_BASE, LDC_B, lsl #2
	sub  x13, x14, LDC_B
	add  x15, x11, LDC_B, lsl #2
	add  x16, x12, LDC_B, lsl #2
	add  x17, x15, LDC_B, lsl #1

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

	stp  q0, q1, [C_M_BASE]
	stp  q2, q3, [x11]
	stp  q4, q5, [x12]
	stp  q6, q7, [x13]
	stp  q8, q25, [x14]
	stp  q26, q27, [x15]
	stp  q28, q29, [x16]
	stp  q30, q31, [x17]
	.endm

// ── m=1/2：只涉及 row0/row1 → v9~v12 ────────────────────────────
//   v9  = [r0c0,r0c1,r1c0,r1c1]
//   v10 = [r0c2,r0c3,r1c2,r1c3]
//   v11 = [r0c4,r0c5,r1c4,r1c5]
//   v12 = [r0c6,r0c7,r1c6,r1c7]

	.macro LOAD_C_2
	add  x11, C_M_BASE, LDC_B          // row1
	ldp  q0, q1, [C_M_BASE]            // row0: col0-3, col4-7
	ldp  q2, q3, [x11]                 // row1: col0-3, col4-7
	zip1 v9.2d,  v0.2d, v2.2d          // [r0c0,r0c1,r1c0,r1c1]
	zip2 v10.2d, v0.2d, v2.2d          // [r0c2,r0c3,r1c2,r1c3]
	zip1 v11.2d, v1.2d, v3.2d          // [r0c4,r0c5,r1c4,r1c5]
	zip2 v12.2d, v1.2d, v3.2d          // [r0c6,r0c7,r1c6,r1c7]
	.endm

	.macro STORE_C_2
	zip1 v0.2d, v9.2d,  v10.2d         // row0: col0-3
	zip1 v1.2d, v11.2d, v12.2d         // row0: col4-7
	zip2 v2.2d, v9.2d,  v10.2d         // row1: col0-3
	zip2 v3.2d, v11.2d, v12.2d         // row1: col4-7
	add  x11, C_M_BASE, LDC_B
	stp  q0, q1, [C_M_BASE]
	stp  q2, q3, [x11]
	.endm

// m=1 时 row1 是无效填充行，只写 row0

	.macro LOAD_C_1
	ldp  q0, q1, [C_M_BASE]            // row0: col0-3, col4-7
	movi v2.16b, #0                    // row1 填零（虚拟行）
	movi v3.16b, #0
	zip1 v9.2d,  v0.2d, v2.2d
	zip2 v10.2d, v0.2d, v2.2d
	zip1 v11.2d, v1.2d, v3.2d
	zip2 v12.2d, v1.2d, v3.2d
	.endm

	.macro STORE_C_1
	zip1 v0.2d, v9.2d,  v10.2d         // row0: col0-3
	zip1 v1.2d, v11.2d, v12.2d         // row0: col4-7
	stp  q0, q1, [C_M_BASE]
	.endm

// ── m=4：row0~row3 → v9~v16 ─────────────────────────────────────

	.macro LOAD_C_4
	add  x11, C_M_BASE, LDC_B
	add  x12, C_M_BASE, LDC_B, lsl #1
	add  x14, C_M_BASE, LDC_B, lsl #2
	sub  x13, x14, LDC_B               // 3*LDC

	ldp  q0, q1, [C_M_BASE]
	ldp  q2, q3, [x11]
	ldp  q4, q5, [x12]
	ldp  q6, q7, [x13]

	zip1 v9.2d,  v0.2d, v2.2d
	zip2 v10.2d, v0.2d, v2.2d
	zip1 v11.2d, v1.2d, v3.2d
	zip2 v12.2d, v1.2d, v3.2d

	zip1 v13.2d, v4.2d, v6.2d
	zip2 v14.2d, v4.2d, v6.2d
	zip1 v15.2d, v5.2d, v7.2d
	zip2 v16.2d, v5.2d, v7.2d
	.endm

	.macro STORE_C_4
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

	stp  q0, q1, [C_M_BASE]
	stp  q2, q3, [x11]
	stp  q4, q5, [x12]
	stp  q6, q7, [x13]
	.endm


	// ── fp32 store variants: LOAD_C_F / STORE_C_F ────────────────────
	// The smmla accumulator is int32 (.4s). These macros add:
	//   LOAD:  fcvtzs (fp32→int32) to load C into accumulators
	//   STORE: scvtf  (int32→fp32) to write accumulators as fp32 output
	// ldc_shift=2, c_adv=32 — same as int32 (both are 4 bytes/element)

	// ── m=8 fp32 ───────────────────────────────────────────────────

		.macro LOAD_C_ROWMAJOR_F
		add  x11, C_M_BASE, LDC_B
		add  x12, C_M_BASE, LDC_B, lsl #1
		add  x14, C_M_BASE, LDC_B, lsl #2
		sub  x13, x14, LDC_B
		add  x15, x11, LDC_B, lsl #2
		add  x16, x12, LDC_B, lsl #2
		add  x17, x15, LDC_B, lsl #1

		ldp  q0, q1, [C_M_BASE]
		ldp  q2, q3, [x11]
		ldp  q4, q5, [x12]
		ldp  q6, q7, [x13]
		ldp  q8, q25, [x14]
		ldp  q26, q27, [x15]
		ldp  q28, q29, [x16]
		ldp  q30, q31, [x17]

		fcvtzs v0.4s, v0.4s
		fcvtzs v1.4s, v1.4s
		fcvtzs v2.4s, v2.4s
		fcvtzs v3.4s, v3.4s
		fcvtzs v4.4s, v4.4s
		fcvtzs v5.4s, v5.4s
		fcvtzs v6.4s, v6.4s
		fcvtzs v7.4s, v7.4s
		fcvtzs v8.4s, v8.4s
		fcvtzs v25.4s, v25.4s
		fcvtzs v26.4s, v26.4s
		fcvtzs v27.4s, v27.4s
		fcvtzs v28.4s, v28.4s
		fcvtzs v29.4s, v29.4s
		fcvtzs v30.4s, v30.4s
		fcvtzs v31.4s, v31.4s

		zip1 v9.2d,  v0.2d, v2.2d
		zip2 v10.2d, v0.2d, v2.2d
		zip1 v11.2d, v1.2d, v3.2d
		zip2 v12.2d, v1.2d, v3.2d

		zip1 v13.2d, v4.2d, v6.2d
		zip2 v14.2d, v4.2d, v6.2d
		zip1 v15.2d, v5.2d, v7.2d
		zip2 v16.2d, v5.2d, v7.2d

		zip1 v17.2d, v8.2d,  v26.2d
		zip2 v18.2d, v8.2d,  v26.2d
		zip1 v19.2d, v25.2d, v27.2d
		zip2 v20.2d, v25.2d, v27.2d

		zip1 v21.2d, v28.2d, v30.2d
		zip2 v22.2d, v28.2d, v30.2d
		zip1 v23.2d, v29.2d, v31.2d
		zip2 v24.2d, v29.2d, v31.2d
		.endm

		.macro STORE_C_ROWMAJOR_F
		add  x11, C_M_BASE, LDC_B
		add  x12, C_M_BASE, LDC_B, lsl #1
		add  x14, C_M_BASE, LDC_B, lsl #2
		sub  x13, x14, LDC_B
		add  x15, x11, LDC_B, lsl #2
		add  x16, x12, LDC_B, lsl #2
		add  x17, x15, LDC_B, lsl #1

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

		stp  q0, q1, [C_M_BASE]
		stp  q2, q3, [x11]
		stp  q4, q5, [x12]
		stp  q6, q7, [x13]
		stp  q8, q25, [x14]
		stp  q26, q27, [x15]
		stp  q28, q29, [x16]
		stp  q30, q31, [x17]
		.endm

	// ── m=1 fp32 ───────────────────────────────────────────────────

		.macro LOAD_C_1_F
		ldp  q0, q1, [C_M_BASE]
		movi v2.16b, #0
		movi v3.16b, #0
		fcvtzs v0.4s, v0.4s
		fcvtzs v1.4s, v1.4s
		zip1 v9.2d,  v0.2d, v2.2d
		zip2 v10.2d, v0.2d, v2.2d
		zip1 v11.2d, v1.2d, v3.2d
		zip2 v12.2d, v1.2d, v3.2d
		.endm

		.macro STORE_C_1_F
		zip1 v0.2d, v9.2d,  v10.2d
		zip1 v1.2d, v11.2d, v12.2d
		scvtf v0.4s, v0.4s
		scvtf v1.4s, v1.4s
		stp  q0, q1, [C_M_BASE]
		.endm

	// ── m=2 fp32 ───────────────────────────────────────────────────

		.macro LOAD_C_2_F
		add  x11, C_M_BASE, LDC_B
		ldp  q0, q1, [C_M_BASE]
		ldp  q2, q3, [x11]
		fcvtzs v0.4s, v0.4s
		fcvtzs v1.4s, v1.4s
		fcvtzs v2.4s, v2.4s
		fcvtzs v3.4s, v3.4s
		zip1 v9.2d,  v0.2d, v2.2d
		zip2 v10.2d, v0.2d, v2.2d
		zip1 v11.2d, v1.2d, v3.2d
		zip2 v12.2d, v1.2d, v3.2d
		.endm

		.macro STORE_C_2_F
		zip1 v0.2d, v9.2d,  v10.2d
		zip1 v1.2d, v11.2d, v12.2d
		zip2 v2.2d, v9.2d,  v10.2d
		zip2 v3.2d, v11.2d, v12.2d
		scvtf v0.4s, v0.4s
		scvtf v1.4s, v1.4s
		scvtf v2.4s, v2.4s
		scvtf v3.4s, v3.4s
		add  x11, C_M_BASE, LDC_B
		stp  q0, q1, [C_M_BASE]
		stp  q2, q3, [x11]
		.endm

	// ── m=4 fp32 ───────────────────────────────────────────────────

		.macro LOAD_C_4_F
		add  x11, C_M_BASE, LDC_B
		add  x12, C_M_BASE, LDC_B, lsl #1
		add  x14, C_M_BASE, LDC_B, lsl #2
		sub  x13, x14, LDC_B

		ldp  q0, q1, [C_M_BASE]
		ldp  q2, q3, [x11]
		ldp  q4, q5, [x12]
		ldp  q6, q7, [x13]

		fcvtzs v0.4s, v0.4s
		fcvtzs v1.4s, v1.4s
		fcvtzs v2.4s, v2.4s
		fcvtzs v3.4s, v3.4s
		fcvtzs v4.4s, v4.4s
		fcvtzs v5.4s, v5.4s
		fcvtzs v6.4s, v6.4s
		fcvtzs v7.4s, v7.4s

		zip1 v9.2d,  v0.2d, v2.2d
		zip2 v10.2d, v0.2d, v2.2d
		zip1 v11.2d, v1.2d, v3.2d
		zip2 v12.2d, v1.2d, v3.2d

		zip1 v13.2d, v4.2d, v6.2d
		zip2 v14.2d, v4.2d, v6.2d
		zip1 v15.2d, v5.2d, v7.2d
		zip2 v16.2d, v5.2d, v7.2d
		.endm

		.macro STORE_C_4_F
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

		stp  q0, q1, [C_M_BASE]
		stp  q2, q3, [x11]
		stp  q4, q5, [x12]
		stp  q6, q7, [x13]
		.endm

// ════════════════════════════════════════════════════════════════════
// CACHED PATH  按 mrows 特化
// 从 A_reorder 连续读取（后续 N iteration 使用）
// ════════════════════════════════════════════════════════════════════

// ── m=8 cached：A 4 regs = 8 rows × 4 cols int8 = 32 bytes ──────
// B 4 regs = 8 rows × 4 cols int8 = 32 bytes

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

// ── m=1/2 cached：A 用 v0,v8（每 K-block 16 bytes = 1 q）───────

	.macro LOAD_A0_B0_2
	ldr  q0, [A_ADDR], #16             // A block0: row0|row1
	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32
	.endm

	.macro COMPUTE_A1_B1_2
	ldr  q8, [A_ADDR], #16             // A block1
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

// ── m=4 cached：A 用 v0~v1, v8,v25（每 K-block 32 bytes = 2 q）──

	.macro LOAD_A0_B0_4
	ldp  q0, q1, [A_ADDR], #32         // A block0: row0|row1, row2|row3
	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32
	.endm

	.macro COMPUTE_A1_B1_4
	ldp  q8, q25, [A_ADDR], #32        // A block1
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

// ════════════════════════════════════════════════════════════════════
// STRIDED PATH  按 mrows 特化
// 从原始 A 跨 LDA 读取，同时写 A_reorder（首个 N iteration 使用）
// ════════════════════════════════════════════════════════════════════

// ── m=8 strided：8 行 × 4 cols int8 = 32 bytes → 4 regs ─────────

	.macro LOAD_A0_B0_STRIDED_8
	add  x11, A_ADDR, LDA                // 1*LDA
	add  x12, A_ADDR, LDA, lsl #1        // 2*LDA
	add  x14, A_ADDR, LDA, lsl #2        // 4*LDA
	sub  x13, x14, LDA                   // 3*LDA = 4-1
	add  x15, x11, LDA, lsl #2           // 5*LDA = 1+4
	add  x16, x12, LDA, lsl #2           // 6*LDA = 2+4
	add  x17, x15, LDA, lsl #1           // 7*LDA = 5+2

	ld1  {v0.d}[0], [A_ADDR]          // row 0 → v0.d[0]
	ld1  {v0.d}[1], [x11]             // row 1 → v0.d[1]
	ld1  {v1.d}[0], [x12]             // row 2 → v1.d[0]
	ld1  {v1.d}[1], [x13]             // row 3 → v1.d[1]
	ld1  {v2.d}[0], [x14]             // row 4 → v2.d[0]
	ld1  {v2.d}[1], [x15]             // row 5 → v2.d[1]
	ld1  {v3.d}[0], [x16]             // row 6 → v3.d[0]
	ld1  {v3.d}[1], [x17]             // row 7 → v3.d[1]

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

// ── m=1 strided：只读 row0，row1 槽填零 ─────────────────────────

	.macro LOAD_A0_B0_STRIDED_1
	ldr  d0, [A_ADDR]                  // row0 → v0.d[0]
	str  q0, [A_REORDER], #16          // 写 A_reorder (upper 64b=0)
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

// ── m=2 strided：读 row0/row1，各取 8 bytes 拼成一个 q ──────────

	.macro LOAD_A0_B0_STRIDED_2
	add  x11, A_ADDR, LDA
	ldr  d0, [A_ADDR]                  // row0 → v0.d[0]
	ld1  {v0.d}[1], [x11]             // row1 → v0.d[1]
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

// ── m=4 strided：读 row0~row3，打包成 2 个 q ────────────────────

	.macro LOAD_A0_B0_STRIDED_4
	add  x11, A_ADDR, LDA
	add  x12, A_ADDR, LDA, lsl #1
	add  x14, A_ADDR, LDA, lsl #2
	sub  x13, x14, LDA                 // 3*LDA

	ldr  d0, [A_ADDR]
	ld1  {v0.d}[1], [x11]             // v0 = row0|row1
	ldr  d1, [x12]
	ld1  {v1.d}[1], [x13]             // v1 = row2|row3

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

// ════════════════════════════════════════════════════════════════════
// GEMM_BODY：统一函数主体宏（零 .if，全参数化）
//   \func_name  : 符号名
//   \mrows      : 1 / 2 / 4 / 8  — M 循环控制 & strided 宏后缀
//   \cmpsfx     : 2 / 4 / 8       — cached COMPUTE/LOAD 宏后缀
//   \load_c     : LOAD_C 宏名
//   \store_c    : STORE_C 宏名
//   \ldc_shift  : ldc 左移位数（2 = *4 bytes）
//   \c_adv      : C 列前进字节数（32 = 8×int32）
// ════════════════════════════════════════════════════════════════════

	.macro GEMM_BODY func_name, mrows, cmpsfx, load_c, store_c, ldc_shift, c_adv, first_n

#ifdef __APPLE__
	.global _\func_name
_\func_name:
#else
	.global \func_name
\func_name:
#endif
	INIT

	// New calling convention: x3 = A_REORDER, x4 = PARAMS
	mov  A_REORDER, x3              // save A_REORDER → x6

	// Load m,k,n,lda,ldb,ldc from params struct (6×int32 = 24 bytes)
	mov  x11, x4                    // keep params pointer before K overwrites x4
	ldp  w3, w4, [x11]              // M, K
	ldp  w5, w7, [x11, #8]          // N, LDA (byte stride, no shift needed)
	ldp  w8, w12, [x11, #16]        // LDB, LDC

	lsl  LDC_B, x12, #\ldc_shift

	mov  A_ORIG_M, A_ORIG
	mov  C_ROW_BASE, C_ADDR_0

	M_LOOP_BEGIN_\mrows

	mov  A_REORDER_M, A_REORDER
	mov  FIRST_N, #\first_n
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

// ── CACHED PATH ───────────────────────────────────────────────────
	mov  A_ADDR, A_REORDER_M

	\load_c

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

// ── STRIDED PATH ──────────────────────────────────────────────────
.L\func_name\()_strided:
	mov  A_ADDR, A_ORIG_M

	\load_c

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

// ── EPILOGUE ──────────────────────────────────────────────────────
.L\func_name\()_epilogue:
	\store_c
	add  C_N_BASE, C_N_BASE, #\c_adv
	add  B_N_BASE, B_N_BASE, LDB, lsl #3

	subs N_COUNT, N_COUNT, #1
	b.gt .L\func_name\()_n_iter

	ADVANCE_PREPACKED_A_\first_n
	M_LOOP_END_\mrows

.L\func_name\()_end:
	RESTORE
	ret

	.endm   // GEMM_BODY

// ════════════════════════════════════════════════════════════════════
// 四个函数实例化：主 m=8 + 尾部 m=1,2,4
// ════════════════════════════════════════════════════════════════════

// 主内核 (m=8)
GEMM_BODY i8gemm_k_ld,  8, 8, LOAD_C_ROWMAJOR, STORE_C_ROWMAJOR, 2, 32, 1

// 尾部内核 (m=1,2,4)
GEMM_BODY i8gemm_k_ld1, 1, 2, LOAD_C_1,        STORE_C_1,        2, 32, 1
GEMM_BODY i8gemm_k_ld2, 2, 2, LOAD_C_2,        STORE_C_2,        2, 32, 1
GEMM_BODY i8gemm_k_ld4, 4, 4, LOAD_C_4,        STORE_C_4,        2, 32, 1

// int32 zero-C variants (skip C load, store int32 row-major)
GEMM_BODY i8gemm_k_zero,  8, 8, LOAD_ZERO_8, STORE_C_ROWMAJOR, 2, 32, 1
GEMM_BODY i8gemm_k_zero1, 1, 2, LOAD_ZERO_1, STORE_C_1,        2, 32, 1
GEMM_BODY i8gemm_k_zero2, 2, 2, LOAD_ZERO_2, STORE_C_2,        2, 32, 1
GEMM_BODY i8gemm_k_zero4, 4, 4, LOAD_ZERO_4, STORE_C_4,        2, 32, 1

// fp32 store variants (_f suffix)
GEMM_BODY i8gemm_k_ld_f,  8, 8, LOAD_C_ROWMAJOR_F, STORE_C_ROWMAJOR_F, 2, 32, 1
GEMM_BODY i8gemm_k_ld1_f, 1, 2, LOAD_C_1_F,        STORE_C_1_F,        2, 32, 1
GEMM_BODY i8gemm_k_ld2_f, 2, 2, LOAD_C_2_F,        STORE_C_2_F,        2, 32, 1
GEMM_BODY i8gemm_k_ld4_f, 4, 4, LOAD_C_4_F,        STORE_C_4_F,        2, 32, 1

// Prepacked-A variants: start directly from cached A_reorder.
GEMM_BODY i8gemm_k_reo_ld,  8, 8, LOAD_C_ROWMAJOR, STORE_C_ROWMAJOR, 2, 32, 0
GEMM_BODY i8gemm_k_reo_ld1, 1, 2, LOAD_C_1,        STORE_C_1,        2, 32, 0
GEMM_BODY i8gemm_k_reo_ld2, 2, 2, LOAD_C_2,        STORE_C_2,        2, 32, 0
GEMM_BODY i8gemm_k_reo_ld4, 4, 4, LOAD_C_4,        STORE_C_4,        2, 32, 0

GEMM_BODY i8gemm_k_reo_zero,  8, 8, LOAD_ZERO_8, STORE_C_ROWMAJOR, 2, 32, 0
GEMM_BODY i8gemm_k_reo_zero1, 1, 2, LOAD_ZERO_1, STORE_C_1,        2, 32, 0
GEMM_BODY i8gemm_k_reo_zero2, 2, 2, LOAD_ZERO_2, STORE_C_2,        2, 32, 0
GEMM_BODY i8gemm_k_reo_zero4, 4, 4, LOAD_ZERO_4, STORE_C_4,        2, 32, 0

GEMM_BODY i8gemm_k_reo_ld_f,  8, 8, LOAD_C_ROWMAJOR_F, STORE_C_ROWMAJOR_F, 2, 32, 0
GEMM_BODY i8gemm_k_reo_ld1_f, 1, 2, LOAD_C_1_F,        STORE_C_1_F,        2, 32, 0
GEMM_BODY i8gemm_k_reo_ld2_f, 2, 2, LOAD_C_2_F,        STORE_C_2_F,        2, 32, 0
GEMM_BODY i8gemm_k_reo_ld4_f, 4, 4, LOAD_C_4_F,        STORE_C_4_F,        2, 32, 0
