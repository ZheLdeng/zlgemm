// bf16gemm_k_ld.S
// void bf16gemm_k_ld(const uint16_t *A,            // x0  原始 A, M×K 行主序 bf16
//                    const uint16_t *B_reordered,   // x1  已 pack 的 B
//                    float         *C,              // x2  行主序输出 float32
//                    int            m,              // x3  M 维度（8 的倍数）
//                    int            k,              // x4  K 维度（8 的倍数）
//                    int            n,              // x5  N 维度（8 的倍数）
//                    uint16_t      *A_reorder,      // x6  A 重排输出 buffer（m*k*2 bytes）
//                    int            lda,            // x7  A 的行间 element stride（大 A 的 K）
//                    int            ldb,            // [sp] K_big of packed B（用于 N-block stride）
//                    int            ldc);           // [sp] C 的 leading dimension（float32 元素数）
//
// 循环顺序: M-N-K (最外层M, 中间N, 最内层K)
// A 重排策略:
//   每个 M block 的第一个 N iteration 从原始 A 跨 stride (lda) 读取，
//   写入 A_reorder; 后续 N iteration 直接从 A_reorder 连续读取。
// B: 预先 pack 好（外部调用者负责），N-block stride = ldb * 16 bytes。
//
// 内核每次处理 8×8 输出块，v9~v24 的 bfmmla 格式：
//   bfmmla computes: 2×4 bf16 (vn.8h) × 4×2 bf16 (vm.8h) → 2×2 f32 accumulate (vd.4s)
//   v9  = C[0:2][0:2], v10 = C[0:2][2:4], v11 = C[0:2][4:6], v12 = C[0:2][6:8]
//   v13-v16: r2/r3 同上, v17-v20: r4/r5, v21-v24: r6/r7

	.text
	.align 4
#ifdef __APPLE__
	.global _bf16gemm_k_ld
#else
	.global bf16gemm_k_ld
#endif

#define A_ORIG      x0
#define B_ADDR      x1
#define C_ADDR_0    x2
#define M           x3
#define K           x4
#define N           x5
#define A_REORDER   x6
#define LDA_B       x7    // A row stride in bytes (= lda * 2, computed in init)

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

#define LDC_B       x9    // C row stride in bytes = ldc * 4
#define LDB         x8    // K_big of packed B (loaded from stack)

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

// ── LOAD_C_ROWMAJOR：从行主序 C 加载 8×8 float32 → blocked v9~v24 ──
// C row stride = LDC_B (= ldc * 4) bytes
	.macro LOAD_C_ROWMAJOR
	// addr: all from C_M_BASE directly, max depth 3
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

	// zip pairs to form 2×2 blocked layout
	zip1 v9.2d,  v0.2d, v2.2d
	zip2 v10.2d, v0.2d, v2.2d
	zip1 v11.2d, v1.2d, v3.2d
	zip2 v12.2d, v1.2d, v3.2d

	zip1 v13.2d, v4.2d, v6.2d
	zip2 v14.2d, v4.2d, v6.2d
	zip1 v15.2d, v5.2d, v7.2d
	zip2 v16.2d, v5.2d, v7.2d

	zip1 v17.2d, v8.2d, v26.2d
	zip2 v18.2d, v8.2d, v26.2d
	zip1 v19.2d, v25.2d, v27.2d
	zip2 v20.2d, v25.2d, v27.2d

	zip1 v21.2d, v28.2d, v30.2d
	zip2 v22.2d, v28.2d, v30.2d
	zip1 v23.2d, v29.2d, v31.2d
	zip2 v24.2d, v29.2d, v31.2d
	.endm

// ── STORE_C_ROWMAJOR：将 blocked v9~v24 按行主序写回 C ────────
	.macro STORE_C_ROWMAJOR
	// addr: all from C_M_BASE directly, max depth 3
	add  x11, C_M_BASE, LDC_B             // 1*LDC
	add  x12, C_M_BASE, LDC_B, lsl #1     // 2*LDC
	add  x14, C_M_BASE, LDC_B, lsl #2     // 4*LDC
	sub  x13, x14, LDC_B                  // 3*LDC = 4-1
	add  x15, x11, LDC_B, lsl #2          // 5*LDC = 1+4
	add  x16, x12, LDC_B, lsl #2          // 6*LDC = 2+4
	add  x17, x15, LDC_B, lsl #1          // 7*LDC = 5+2

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

// ==================== CACHED PATH MACROS ====================
// 从 A_reorder 连续读取 A（后续 N iteration 使用）
//
// A 加载: 4 regs = 8 rows × 4 cols bf16 = 64 bytes
//   v0 = A[0:2][k:k+4], v1 = A[2:4][k:k+4]
//   v2 = A[4:6][k:k+4], v3 = A[6:8][k:k+4]
// B 加载: 4 regs = 4 rows × 8 cols bf16 = 64 bytes
//   v4 = B[k:k+4][0:2], v5 = B[k:k+4][2:4]
//   v6 = B[k:k+4][4:6], v7 = B[k:k+4][6:8]

	.macro LOAD_A0_B0
	ldp q0, q1, [A_ADDR], #32
	ldp q2, q3, [A_ADDR], #32
	ldp q4, q5, [B_ADDR_1], #32
	ldp q6, q7, [B_ADDR_1], #32
	.endm

	.macro COMPUTE_A1_B1
	// Load next A and B (prefetch)
	ldp q8,  q25, [A_ADDR], #32
	ldp q26, q27, [A_ADDR], #32
	ldp q28, q29, [B_ADDR_1], #32
	ldp q30, q31, [B_ADDR_1], #32

	// Compute 16 bfmmla: A0×B0 (v0-v3 × v4-v7)
	bfmmla v9.4s,  v0.8h, v4.8h
	bfmmla v10.4s, v0.8h, v5.8h
	bfmmla v11.4s, v0.8h, v6.8h
	bfmmla v12.4s, v0.8h, v7.8h

	bfmmla v13.4s, v1.8h, v4.8h
	bfmmla v14.4s, v1.8h, v5.8h
	bfmmla v15.4s, v1.8h, v6.8h
	bfmmla v16.4s, v1.8h, v7.8h

	bfmmla v17.4s, v2.8h, v4.8h
	bfmmla v18.4s, v2.8h, v5.8h
	bfmmla v19.4s, v2.8h, v6.8h
	bfmmla v20.4s, v2.8h, v7.8h

	bfmmla v21.4s, v3.8h, v4.8h
	bfmmla v22.4s, v3.8h, v5.8h
	bfmmla v23.4s, v3.8h, v6.8h
	bfmmla v24.4s, v3.8h, v7.8h
	.endm

	.macro COMPUTE_A2_B2
	// Load next A and B (prefetch)
	ldp q0, q1, [A_ADDR], #32
	ldp q2, q3, [A_ADDR], #32
	ldp q4, q5, [B_ADDR_1], #32
	ldp q6, q7, [B_ADDR_1], #32

	// Compute 16 bfmmla: A1×B1 (v8,v25-v27 × v28-v31)
	bfmmla v9.4s,  v8.8h, v28.8h
	bfmmla v10.4s, v8.8h, v29.8h
	bfmmla v11.4s, v8.8h, v30.8h
	bfmmla v12.4s, v8.8h, v31.8h

	bfmmla v13.4s, v25.8h, v28.8h
	bfmmla v14.4s, v25.8h, v29.8h
	bfmmla v15.4s, v25.8h, v30.8h
	bfmmla v16.4s, v25.8h, v31.8h

	bfmmla v17.4s, v26.8h, v28.8h
	bfmmla v18.4s, v26.8h, v29.8h
	bfmmla v19.4s, v26.8h, v30.8h
	bfmmla v20.4s, v26.8h, v31.8h

	bfmmla v21.4s, v27.8h, v28.8h
	bfmmla v22.4s, v27.8h, v29.8h
	bfmmla v23.4s, v27.8h, v30.8h
	bfmmla v24.4s, v27.8h, v31.8h
	.endm

	.macro COMPUTE_A3_B3
	// Compute with last A,B pair (no prefetch)
	bfmmla v9.4s,  v8.8h, v28.8h
	bfmmla v10.4s, v8.8h, v29.8h
	bfmmla v11.4s, v8.8h, v30.8h
	bfmmla v12.4s, v8.8h, v31.8h

	bfmmla v13.4s, v25.8h, v28.8h
	bfmmla v14.4s, v25.8h, v29.8h
	bfmmla v15.4s, v25.8h, v30.8h
	bfmmla v16.4s, v25.8h, v31.8h

	bfmmla v17.4s, v26.8h, v28.8h
	bfmmla v18.4s, v26.8h, v29.8h
	bfmmla v19.4s, v26.8h, v30.8h
	bfmmla v20.4s, v26.8h, v31.8h

	bfmmla v21.4s, v27.8h, v28.8h
	bfmmla v22.4s, v27.8h, v29.8h
	bfmmla v23.4s, v27.8h, v30.8h
	bfmmla v24.4s, v27.8h, v31.8h
	.endm

// ==================== STRIDED PATH MACROS ====================
// 从原始 A 跨 stride (LDA_B) 读取，写入 A_reorder（首个 N iteration 使用）
// A 每行加载 4 bf16 = 8 bytes → 8 行 = 64 bytes = 4 NEON regs

	.macro LOAD_A0_B0_STRIDED
	// ── 从原始 A 跨 stride 加载 A0 → v0-v3 ──
	// 预计算 8 个行地址，row 0 直接用 A_ADDR，最大深度 3
	add  x11, A_ADDR, LDA_B                // 1*LDA
	add  x12, A_ADDR, LDA_B, lsl #1        // 2*LDA
	add  x14, A_ADDR, LDA_B, lsl #2        // 4*LDA
	sub  x13, x14, LDA_B                   // 3*LDA = 4-1
	add  x15, x11, LDA_B, lsl #2           // 5*LDA = 1+4
	add  x16, x12, LDA_B, lsl #2           // 6*LDA = 2+4
	add  x17, x15, LDA_B, lsl #1           // 7*LDA = 5+2

	// 8 路独立发射，无寄存器依赖
	// 每行加载 4 bf16 = 8 bytes = 1 doubleword
	ld1  {v0.d}[0], [A_ADDR]           // row 0 → v0.d[0]
	ld1  {v0.d}[1], [x11]              // row 1 → v0.d[1]
	ld1  {v1.d}[0], [x12]              // row 2 → v1.d[0]
	ld1  {v1.d}[1], [x13]              // row 3 → v1.d[1]
	ld1  {v2.d}[0], [x14]              // row 4 → v2.d[0]
	ld1  {v2.d}[1], [x15]              // row 5 → v2.d[1]
	ld1  {v3.d}[0], [x16]              // row 6 → v3.d[0]
	ld1  {v3.d}[1], [x17]              // row 7 → v3.d[1]

	// ── 写入 A_reorder（连续 64 字节）──
	stp  q0, q1, [A_REORDER], #32
	stp  q2, q3, [A_REORDER], #32

	// ── A_ADDR 前进到下一个 K block（原始 A 中 +4 列 = +8 bytes）──
	add  A_ADDR, A_ADDR, #8

	// ── 加载 B0（同 cached path）──
	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32
	.endm

	.macro COMPUTE_A1_B1_STRIDED
	// ── 从原始 A 跨 stride 加载 A1 → v8, v25-v27 ──
	add  x11, A_ADDR, LDA_B                // 1*LDA
	add  x12, A_ADDR, LDA_B, lsl #1        // 2*LDA
	add  x14, A_ADDR, LDA_B, lsl #2        // 4*LDA
	sub  x13, x14, LDA_B                   // 3*LDA = 4-1
	add  x15, x11, LDA_B, lsl #2           // 5*LDA = 1+4
	add  x16, x12, LDA_B, lsl #2           // 6*LDA = 2+4
	add  x17, x15, LDA_B, lsl #1           // 7*LDA = 5+2

	ld1  {v8.d}[0],  [A_ADDR]          // row 0 → v8.d[0]
	ld1  {v8.d}[1],  [x11]             // row 1 → v8.d[1]
	ld1  {v25.d}[0], [x12]             // row 2 → v25.d[0]
	ld1  {v25.d}[1], [x13]             // row 3 → v25.d[1]
	ld1  {v26.d}[0], [x14]             // row 4 → v26.d[0]
	ld1  {v26.d}[1], [x15]             // row 5 → v26.d[1]
	ld1  {v27.d}[0], [x16]             // row 6 → v27.d[0]
	ld1  {v27.d}[1], [x17]             // row 7 → v27.d[1]

	// ── 写入 A_reorder ──
	stp  q8,  q25, [A_REORDER], #32
	stp  q26, q27, [A_REORDER], #32

	// ── A_ADDR 前进到下一个 K block ──
	add  A_ADDR, A_ADDR, #8

	// ── 加载 B1 ──
	ldp  q28, q29, [B_ADDR_1], #32
	ldp  q30, q31, [B_ADDR_1], #32

	// ── 计算 A0 × B0 ──
	bfmmla v9.4s,  v0.8h, v4.8h
	bfmmla v10.4s, v0.8h, v5.8h
	bfmmla v11.4s, v0.8h, v6.8h
	bfmmla v12.4s, v0.8h, v7.8h

	bfmmla v13.4s, v1.8h, v4.8h
	bfmmla v14.4s, v1.8h, v5.8h
	bfmmla v15.4s, v1.8h, v6.8h
	bfmmla v16.4s, v1.8h, v7.8h

	bfmmla v17.4s, v2.8h, v4.8h
	bfmmla v18.4s, v2.8h, v5.8h
	bfmmla v19.4s, v2.8h, v6.8h
	bfmmla v20.4s, v2.8h, v7.8h

	bfmmla v21.4s, v3.8h, v4.8h
	bfmmla v22.4s, v3.8h, v5.8h
	bfmmla v23.4s, v3.8h, v6.8h
	bfmmla v24.4s, v3.8h, v7.8h
	.endm

	.macro COMPUTE_A2_B2_STRIDED
	// ── 从原始 A 跨 stride 加载 A2 → v0-v3 ──
	add  x11, A_ADDR, LDA_B                // 1*LDA
	add  x12, A_ADDR, LDA_B, lsl #1        // 2*LDA
	add  x14, A_ADDR, LDA_B, lsl #2        // 4*LDA
	sub  x13, x14, LDA_B                   // 3*LDA = 4-1
	add  x15, x11, LDA_B, lsl #2           // 5*LDA = 1+4
	add  x16, x12, LDA_B, lsl #2           // 6*LDA = 2+4
	add  x17, x15, LDA_B, lsl #1           // 7*LDA = 5+2

	ld1  {v0.d}[0], [A_ADDR]           // row 0 → v0.d[0]
	ld1  {v0.d}[1], [x11]              // row 1 → v0.d[1]
	ld1  {v1.d}[0], [x12]              // row 2 → v1.d[0]
	ld1  {v1.d}[1], [x13]              // row 3 → v1.d[1]
	ld1  {v2.d}[0], [x14]              // row 4 → v2.d[0]
	ld1  {v2.d}[1], [x15]              // row 5 → v2.d[1]
	ld1  {v3.d}[0], [x16]              // row 6 → v3.d[0]
	ld1  {v3.d}[1], [x17]              // row 7 → v3.d[1]

	// ── 写入 A_reorder ──
	stp  q0, q1, [A_REORDER], #32
	stp  q2, q3, [A_REORDER], #32

	// ── A_ADDR 前进到下一个 K block ──
	add  A_ADDR, A_ADDR, #8

	// ── 加载 B2 ──
	ldp  q4, q5, [B_ADDR_1], #32
	ldp  q6, q7, [B_ADDR_1], #32

	// ── 计算 A1 × B1 ──
	bfmmla v9.4s,  v8.8h, v28.8h
	bfmmla v10.4s, v8.8h, v29.8h
	bfmmla v11.4s, v8.8h, v30.8h
	bfmmla v12.4s, v8.8h, v31.8h

	bfmmla v13.4s, v25.8h, v28.8h
	bfmmla v14.4s, v25.8h, v29.8h
	bfmmla v15.4s, v25.8h, v30.8h
	bfmmla v16.4s, v25.8h, v31.8h

	bfmmla v17.4s, v26.8h, v28.8h
	bfmmla v18.4s, v26.8h, v29.8h
	bfmmla v19.4s, v26.8h, v30.8h
	bfmmla v20.4s, v26.8h, v31.8h

	bfmmla v21.4s, v27.8h, v28.8h
	bfmmla v22.4s, v27.8h, v29.8h
	bfmmla v23.4s, v27.8h, v30.8h
	bfmmla v24.4s, v27.8h, v31.8h
	.endm

// ==================== 主入口 ====================

#ifdef __APPLE__
_bf16gemm_k_ld:
#else
bf16gemm_k_ld:
#endif
	INIT

	// ── 从栈加载 ldb, ldc ──
	// INIT 保存了 10 个 stp pair（160 bytes）
#ifdef __APPLE__
	// macOS ARM64: 32-bit stack args packed at 4-byte intervals
	ldr  w8,  [sp, #160]         // ldb → w8 (zero-extends to x8 = LDB)
	ldr  w12, [sp, #164]         // ldc → w12 (temp)
#else
	// Linux AAPCS64: 每个栈参数占 8 字节（64-bit aligned）
	ldr  w8,  [sp, #160]         // ldb → w8 (zero-extends to x8 = LDB)
	ldr  w12, [sp, #168]         // ldc → w12 (temp)
#endif
	lsl  LDC_B, x12, #2          // LDC_B = ldc * 4 (bytes per float32 row)

	// ── 将 LDA 从 element count 转换为 byte stride ──
	// lda 参数为 bf16 element 数，每 element 2 bytes
	lsl  LDA_B, LDA_B, #1        // LDA_B = lda * 2 (bytes)

	mov  A_ORIG_M, A_ORIG       // A_ORIG_M: 原始 A 中当前 M block 的基地址
	mov  C_ROW_BASE, C_ADDR_0   // C row base walks forward through M rows

// ── M 方向外层循环（每次 8 行）──────────────────────────────────
401:
	asr  M_COUNT, M, #3
	cmp  M_COUNT, #0
	b.eq 301f

400:
	// 每个 M block 的初始化
	mov  A_REORDER_M, A_REORDER  // 记录此 M block 在 A_reorder 中的起始位置
	mov  FIRST_N, #1             // 标记为首个 N iteration
	mov  C_N_BASE, C_ROW_BASE    // C column base for this M row
	mov  B_N_BASE, B_ADDR        // reset B to origin for each M block

	// ── N 方向内层循环（每次 8 列）──────────────────────────────
	asr  N_COUNT, N, #3
	cmp  N_COUNT, #0
	b.eq 301f

102:
	mov  B_ADDR_1, B_N_BASE      // B pointer for this N panel
	mov  C_M_BASE, C_N_BASE      // C base for this (m,n) tile

	// ── 判断是否首个 N iteration ──
	cmp  FIRST_N, #1
	b.eq 801f                     // 首个 N: 走 strided 路径

// ==================== CACHED PATH ====================
	// 从 A_reorder 连续读取（后续 N iteration）
	mov  A_ADDR, A_REORDER_M     // A_ADDR = 此 M block 在 reorder 中的起始

	LOAD_C_ROWMAJOR

	asr  K_COUNT, K, #3          // K/8 iterations (each processes 2×K_unroll=4)
	cmp  K_COUNT, #0
	b.eq 301f
	subs K_COUNT, K_COUNT, #1
	LOAD_A0_B0
	b.eq 202f

201:
	COMPUTE_A1_B1
	COMPUTE_A2_B2
	subs K_COUNT, K_COUNT, #1
	b.gt 201b
202:
	COMPUTE_A1_B1
	COMPUTE_A3_B3

	b 902f                        // 跳过 strided 路径

// ==================== STRIDED PATH ====================
801:
	// 从原始 A 跨 stride (LDA_B) 读取，写入 A_reorder（首个 N iteration）
	mov  A_ADDR, A_ORIG_M        // A_ADDR = 此 M block 在原始 A 中的起始

	LOAD_C_ROWMAJOR

	asr  K_COUNT, K, #3          // K/8 iterations
	cmp  K_COUNT, #0
	b.eq 301f
	subs K_COUNT, K_COUNT, #1
	LOAD_A0_B0_STRIDED
	b.eq 812f

811:
	COMPUTE_A1_B1_STRIDED
	COMPUTE_A2_B2_STRIDED
	subs K_COUNT, K_COUNT, #1
	b.gt 811b
812:
	COMPUTE_A1_B1_STRIDED
	COMPUTE_A3_B3

	mov  FIRST_N, #0              // 清除首 N 标记

// ==================== COMMON EPILOGUE ====================
902:
	STORE_C_ROWMAJOR

	add  C_N_BASE, C_N_BASE, #32                    // next N column in C (8 float32 = 32 bytes)
	add  B_N_BASE, B_N_BASE, LDB, lsl #4            // next N panel in B: += ldb * 16

	subs N_COUNT, N_COUNT, #1
	b.gt 102b

	// ── 前进到下一个 M block ──
	add  A_ORIG_M, A_ORIG_M, LDA_B, lsl #3          // 原始 A: 跳过 8 行，stride = lda * 8 bytes
	add  C_ROW_BASE, C_ROW_BASE, LDC_B, lsl #3      // C: 跳过 8 行，stride = ldc * 4 * 8 bytes

	subs M_COUNT, M_COUNT, #1
	b.gt 400b

301:
	RESTORE
	ret
