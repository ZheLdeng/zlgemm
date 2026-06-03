// i8gemm_k_ld.S
// void i8gemm_k_ld(const int8_t  *A,            // x0  原始 A, M×K 行主序（大矩阵的子区域）
//                  const int8_t  *B_reordered,  // x1  已 pack 的 B（大 packed B 的子区域）
//                  int32_t       *C,            // x2  行主序输出（大矩阵的子区域）
//                  int            m,            // x3  M 维度（8 的倍数）
//                  int            k,            // x4  K 维度（16 的倍数）
//                  int            n,            // x5  N 维度（8 的倍数）
//                  int8_t        *A_reorder,    // x6  A 重排输出 buffer（m*k bytes）
//                  int            lda,          // x7  A 的行间 byte stride（大 A 的 K, int8=1 byte）
//                  int            ldb,          // [sp] K_big of packed B（用于 N-block stride）
//                  int            ldc);         // [sp] C 的 leading dimension（int32 元素数）
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
#ifdef __APPLE__
.global _i8gemm_k_ld
#else
.global i8gemm_k_ld
#endif

#define A_ORIG      x0
#define B_ADDR      x1
#define C_ADDR_0    x2
#define M           x3
#define K           x4
#define N           x5
#define A_REORDER   x6
#define LDA         x7    // A row stride in bytes

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

// ── LOAD_C_ROWMAJOR：从行主序 C 加载 8×8 块 → blocked v9~v24 ──
// C row stride = LDC_B (= ldc * 4) bytes
// 使用临时寄存器 x11~x17（caller-saved，无需保存）
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

    zip1 v9.2d,  v0.2d, v2.2d
    zip2 v10.2d, v0.2d, v2.2d
    zip1 v11.2d, v1.2d, v3.2d
    zip2 v12.2d, v1.2d, v3.2d

    zip1 v13.2d, v4.2d, v6.2d
    zip2 v14.2d, v4.2d, v6.2d
    zip1 v15.2d, v5.2d, v7.2d
    zip2 v16.2d, v5.2d, v7.2d

    zip1 v17.2d, v8.2d, v10.2d
    zip2 v18.2d, v8.2d, v10.2d
    zip1 v19.2d, v9.2d, v11.2d
    zip2 v20.2d, v9.2d, v11.2d

    zip1 v21.2d, v12.2d, v14.2d
    zip2 v22.2d, v12.2d, v14.2d
    zip1 v23.2d, v13.2d, v15.2d
    zip2 v24.2d, v13.2d, v15.2d

.endm

// ── STORE_C_ROWMAJOR：将 blocked v9~v24 按行主序写回 C ────────
// C row stride = LDC_B (= ldc * 4) bytes
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

.macro LOAD_A0_B0
    ldp q0, q1, [A_ADDR], #32
    ldp q2, q3, [A_ADDR], #32
    ldp q4, q5, [B_ADDR_1], #32
    ldp q6, q7, [B_ADDR_1], #32
.endm

.macro COMPUTE_A1_B1
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

.macro COMPUTE_A2_B2
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

.macro COMPUTE_A3_B3
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

// ==================== STRIDED PATH MACROS ====================
// 从原始 A 跨 stride (LDA) 读取，写入 A_reorder（首个 N iteration 使用）

.macro LOAD_A0_B0_STRIDED
    // ── 从原始 A 跨 stride 加载 A0 → v0-v3 ──
    // 预计算 7 个行地址，row 0 直接用 A_ADDR，最大深度 3
    add  x11, A_ADDR, LDA                // 1*LDA
    add  x12, A_ADDR, LDA, lsl #1        // 2*LDA
    add  x14, A_ADDR, LDA, lsl #2        // 4*LDA
    sub  x13, x14, LDA                   // 3*LDA = 4-1
    add  x15, x11, LDA, lsl #2           // 5*LDA = 1+4
    add  x16, x12, LDA, lsl #2           // 6*LDA = 2+4
    add  x17, x15, LDA, lsl #1           // 7*LDA = 5+2

    // 8 路独立发射，无寄存器依赖
    ld1  {v0.d}[0], [A_ADDR]          // row 0 → v0.d[0]
    ld1  {v0.d}[1], [x11]         // row 1 → v0.d[1]
    ld1  {v1.d}[0], [x12]         // row 2 → v1.d[0]
    ld1  {v1.d}[1], [x13]         // row 3 → v1.d[1]
    ld1  {v2.d}[0], [x14]         // row 4 → v2.d[0]
    ld1  {v2.d}[1], [x15]         // row 5 → v2.d[1]
    ld1  {v3.d}[0], [x16]         // row 6 → v3.d[0]
    ld1  {v3.d}[1], [x17]         // row 7 → v3.d[1]

    // ── 写入 A_reorder（连续 64 字节）──
    stp  q0, q1, [A_REORDER], #32
    stp  q2, q3, [A_REORDER], #32

    // ── A_ADDR 前进到下一个 K block（原始 A 中 +8 列）──
    add  A_ADDR, A_ADDR, #8

    // ── 加载 B0（同 cached path）──
    ldp  q4, q5, [B_ADDR_1], #32
    ldp  q6, q7, [B_ADDR_1], #32
.endm

.macro COMPUTE_A1_B1_STRIDED
    // ── 从原始 A 跨 stride 加载 A1 → v8, v25-v27 ──
    add  x11, A_ADDR, LDA                // 1*LDA
    add  x12, A_ADDR, LDA, lsl #1        // 2*LDA
    add  x14, A_ADDR, LDA, lsl #2        // 4*LDA
    sub  x13, x14, LDA                   // 3*LDA = 4-1
    add  x15, x11, LDA, lsl #2           // 5*LDA = 1+4
    add  x16, x12, LDA, lsl #2           // 6*LDA = 2+4
    add  x17, x15, LDA, lsl #1           // 7*LDA = 5+2

    ld1  {v8.d}[0],  [A_ADDR]        // row 0 → v8.d[0]
    ld1  {v8.d}[1],  [x11]       // row 1 → v8.d[1]
    ld1  {v25.d}[0], [x12]       // row 2 → v25.d[0]
    ld1  {v25.d}[1], [x13]       // row 3 → v25.d[1]
    ld1  {v26.d}[0], [x14]       // row 4 → v26.d[0]
    ld1  {v26.d}[1], [x15]       // row 5 → v26.d[1]
    ld1  {v27.d}[0], [x16]       // row 6 → v27.d[0]
    ld1  {v27.d}[1], [x17]       // row 7 → v27.d[1]

    // ── 写入 A_reorder ──
    stp  q8,  q25, [A_REORDER], #32
    stp  q26, q27, [A_REORDER], #32

    // ── A_ADDR 前进到下一个 K block ──
    add  A_ADDR, A_ADDR, #8

    // ── 加载 B1 ──
    ldp  q28, q29, [B_ADDR_1], #32
    ldp  q30, q31, [B_ADDR_1], #32

    // ── 计算 A0 × B0 ──
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

.macro COMPUTE_A2_B2_STRIDED
    // ── 从原始 A 跨 stride 加载 A2 → v0-v3 ──
    add  x11, A_ADDR, LDA                // 1*LDA
    add  x12, A_ADDR, LDA, lsl #1        // 2*LDA
    add  x14, A_ADDR, LDA, lsl #2        // 4*LDA
    sub  x13, x14, LDA                   // 3*LDA = 4-1
    add  x15, x11, LDA, lsl #2           // 5*LDA = 1+4
    add  x16, x12, LDA, lsl #2           // 6*LDA = 2+4
    add  x17, x15, LDA, lsl #1           // 7*LDA = 5+2

    ld1  {v0.d}[0], [A_ADDR]         // row 0 → v0.d[0]
    ld1  {v0.d}[1], [x11]        // row 1 → v0.d[1]
    ld1  {v1.d}[0], [x12]        // row 2 → v1.d[0]
    ld1  {v1.d}[1], [x13]        // row 3 → v1.d[1]
    ld1  {v2.d}[0], [x14]        // row 4 → v2.d[0]
    ld1  {v2.d}[1], [x15]        // row 5 → v2.d[1]
    ld1  {v3.d}[0], [x16]        // row 6 → v3.d[0]
    ld1  {v3.d}[1], [x17]        // row 7 → v3.d[1]

    // ── 写入 A_reorder ──
    stp  q0, q1, [A_REORDER], #32
    stp  q2, q3, [A_REORDER], #32

    // ── A_ADDR 前进到下一个 K block ──
    add  A_ADDR, A_ADDR, #8

    // ── 加载 B2 ──
    ldp  q4, q5, [B_ADDR_1], #32
    ldp  q6, q7, [B_ADDR_1], #32

    // ── 计算 A1 × B1 ──
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

// ==================== 主入口 ====================

#ifdef __APPLE__
_i8gemm_k_ld:
#else
i8gemm_k_ld:
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
    lsl  LDC_B, x12, #2          // LDC_B = ldc * 4 (bytes)

    // LDA 已在 x7，无需加载

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

101:
    asr  N_COUNT, N, #3
    cmp  N_COUNT, #0
    b.eq 301f

// ── N 方向内层循环（每次 8 列）──────────────────────────────────
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

    asr  K_COUNT, K, #4
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
    // 从原始 A 跨 stride (LDA) 读取，写入 A_reorder（首个 N iteration）
    mov  A_ADDR, A_ORIG_M        // A_ADDR = 此 M block 在原始 A 中的起始

    LOAD_C_ROWMAJOR

    asr  K_COUNT, K, #4
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

    add  C_N_BASE, C_N_BASE, #32                    // next N column in C
    add  B_N_BASE, B_N_BASE, LDB, lsl #3            // next N panel in B: += ldb * 8

    subs N_COUNT, N_COUNT, #1
    b.gt 102b

    // ── 前进到下一个 M block ──
    add  A_ORIG_M, A_ORIG_M, LDA, lsl #3            // 原始 A: 跳过 8 行，stride = lda
    add  C_ROW_BASE, C_ROW_BASE, LDC_B, lsl #3      // C: 跳过 8 行，stride = ldc * 4

    subs M_COUNT, M_COUNT, #1
    b.gt 400b

301:
    RESTORE
    ret

// ═══════════════════════════════════════════════════════════════
// Tail kernels (m=1,2,4) — appended from i8gemm_k_tail.S
// Note: INIT/RESTORE macros reused from main kernel above
// ═══════════════════════════════════════════════════════════════

.text
.align 4

// ── 寄存器别名（与主文件一致）────────────────────────────────────
#define A_ORIG      x0
#define B_ADDR      x1
#define C_ADDR_0    x2
#define M           x3
#define K           x4
#define N           x5
#define A_REORDER   x6
#define LDA         x7

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

#define LDC_B       x9
#define LDB         x8


// ════════════════════════════════════════════════════════════════════
// LOAD_C / STORE_C  按行数特化
// ════════════════════════════════════════════════════════════════════

// ── m=1/2：只涉及 row0/row1 → v9~v12 ────────────────────────────
// C 布局 [col0..3 | col4..7]，每行 8×int32 = 32 bytes
//   v9  = [r0c0,r0c1,r1c0,r1c1]
//   v10 = [r0c2,r0c3,r1c2,r1c3]
//   v11 = [r0c4,r0c5,r1c4,r1c5]
//   v12 = [r0c6,r0c7,r1c6,r1c7]

.macro LOAD_C_2
    // 读 row0, row1
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

// m=1 时 row1 是无效填充行，只写 row0（取 zip 结果的低64位即可）
// 用 str q（写整个 q 的 low/high 拆分）→ 直接写低 64 位两次更麻烦，
// 改用 zip 后只 store row0 那两个 q

.macro LOAD_C_1
    // 只读 row0；row1 的寄存器置零（供 smmla 正确填充 r1 槽位，结果不写回）
    ldp  q0, q1, [C_M_BASE]            // row0: col0-3, col4-7
    movi v2.16b, #0                    // row1 填零（虚拟行）
    movi v3.16b, #0
    zip1 v9.2d,  v0.2d, v2.2d
    zip2 v10.2d, v0.2d, v2.2d
    zip1 v11.2d, v1.2d, v3.2d
    zip2 v12.2d, v1.2d, v3.2d
.endm

.macro STORE_C_1
    // 只写 row0
    zip1 v0.2d, v9.2d,  v10.2d         // row0: col0-3
    zip1 v1.2d, v11.2d, v12.2d         // row0: col4-7
    stp  q0, q1, [C_M_BASE]
.endm

// ── m=4：row0~row3 → v9~v16 ─────────────────────────────────────
//   v9~v12 : row0/row1（同 m=2）
//   v13~v16: row2/row3
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

// ════════════════════════════════════════════════════════════════════
// CACHED PATH  按 mrows 特化
// A_reorder 中数据已经连续，每 K-block：
//   m=1/2 : 16 bytes（1个 q）
//   m=4   : 32 bytes（2个 q）
// 每次处理 2个 K-block（= 32 bytes A / m=1/2，64 bytes / m=4）
// ════════════════════════════════════════════════════════════════════

// ── m=1/2 cached：A 用 v0,v8；B 与 m=8 相同 ──────────────────────

.macro LOAD_A0_B0_2
    ldr  q0, [A_ADDR], #16             // A block0: row0|row1 (or row0|zero)
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

// ── m=4 cached：A 用 v0~v1, v8,v25；B 与 m=8 相同 ────────────────

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
// 从原始 A 跨 LDA 读取，同时写 A_reorder
// ════════════════════════════════════════════════════════════════════

// ── m=1 strided：只读 row0，row1 槽填零 ─────────────────────────

.macro LOAD_A0_B0_STRIDED_1
    ldr  d0, [A_ADDR]                  // row0 → v0.d[0] (ldr d zero-extends to 128b)
    str  q0, [A_REORDER], #16          // 写 A_reorder (upper 64b=0)
    add  A_ADDR, A_ADDR, #8            // 前进到下一 K-block

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

// ── m=2 strided：读 row0/row1，各取 8 字节拼成一个 q ────────────

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
// GEMM_BODY：函数主体宏
//   \func_name : 符号名（不含平台前缀，由宏内部处理）
//   \mrows     : 1 / 2 / 4
// ════════════════════════════════════════════════════════════════════

.macro GEMM_BODY func_name, mrows

#ifdef __APPLE__
.global _\func_name
_\func_name:
#else
.global \func_name
\func_name:
#endif
    INIT

#ifdef __APPLE__
    ldr  w8,  [sp, #160]
    ldr  w12, [sp, #164]
#else
    ldr  w8,  [sp, #160]
    ldr  w12, [sp, #168]
#endif
    lsl  LDC_B, x12, #2

    mov  A_ORIG_M, A_ORIG
    mov  C_ROW_BASE, C_ADDR_0

// ── 这三个尾部函数 m 固定为 \mrows，没有 M-block 外层循环 ────────
// （调用方保证传入的 m == \mrows，直接执行单 M-block）

    mov  A_REORDER_M, A_REORDER
    mov  FIRST_N, #1
    mov  C_N_BASE, C_ROW_BASE
    mov  B_N_BASE, B_ADDR

// ── N 循环 ────────────────────────────────────────────────────────
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

    .if \mrows == 1
    LOAD_C_1
    .elseif \mrows == 2
    LOAD_C_2
    .else
    LOAD_C_4
    .endif

    asr  K_COUNT, K, #4
    cmp  K_COUNT, #0
    b.eq .L\func_name\()_end
    subs K_COUNT, K_COUNT, #1

    .if \mrows <= 2
    LOAD_A0_B0_2
    .else
    LOAD_A0_B0_4
    .endif

    b.eq .L\func_name\()_cached_tail

.L\func_name\()_cached_loop:
    .if \mrows <= 2
    COMPUTE_A1_B1_2
    COMPUTE_A2_B2_2
    .else
    COMPUTE_A1_B1_4
    COMPUTE_A2_B2_4
    .endif
    subs K_COUNT, K_COUNT, #1
    b.gt .L\func_name\()_cached_loop

.L\func_name\()_cached_tail:
    .if \mrows <= 2
    COMPUTE_A1_B1_2
    COMPUTE_A3_B3_2
    .else
    COMPUTE_A1_B1_4
    COMPUTE_A3_B3_4
    .endif

    b .L\func_name\()_epilogue

// ── STRIDED PATH ──────────────────────────────────────────────────
.L\func_name\()_strided:
    mov  A_ADDR, A_ORIG_M

    .if \mrows == 1
    LOAD_C_1
    .elseif \mrows == 2
    LOAD_C_2
    .else
    LOAD_C_4
    .endif

    asr  K_COUNT, K, #4
    cmp  K_COUNT, #0
    b.eq .L\func_name\()_end
    subs K_COUNT, K_COUNT, #1

    .if \mrows == 1
    LOAD_A0_B0_STRIDED_1
    .elseif \mrows == 2
    LOAD_A0_B0_STRIDED_2
    .else
    LOAD_A0_B0_STRIDED_4
    .endif

    b.eq .L\func_name\()_strided_tail

.L\func_name\()_strided_loop:
    .if \mrows == 1
    COMPUTE_A1_B1_STRIDED_1
    COMPUTE_A2_B2_STRIDED_1
    .elseif \mrows == 2
    COMPUTE_A1_B1_STRIDED_2
    COMPUTE_A2_B2_STRIDED_2
    .else
    COMPUTE_A1_B1_STRIDED_4
    COMPUTE_A2_B2_STRIDED_4
    .endif
    subs K_COUNT, K_COUNT, #1
    b.gt .L\func_name\()_strided_loop

.L\func_name\()_strided_tail:
    .if \mrows == 1
    COMPUTE_A1_B1_STRIDED_1
    COMPUTE_A3_B3_2
    .elseif \mrows == 2
    COMPUTE_A1_B1_STRIDED_2
    COMPUTE_A3_B3_2
    .else
    COMPUTE_A1_B1_STRIDED_4
    COMPUTE_A3_B3_4
    .endif

    mov  FIRST_N, #0

// ── EPILOGUE（每个 N tile 公共）──────────────────────────────────
.L\func_name\()_epilogue:
    .if \mrows == 1
    STORE_C_1
    .elseif \mrows == 2
    STORE_C_2
    .else
    STORE_C_4
    .endif

    add  C_N_BASE, C_N_BASE, #32
    add  B_N_BASE, B_N_BASE, LDB, lsl #3

    subs N_COUNT, N_COUNT, #1
    b.gt .L\func_name\()_n_iter

.L\func_name\()_end:
    RESTORE
    ret

.endm   // GEMM_BODY

// ════════════════════════════════════════════════════════════════════
// 三个函数实例化
// ════════════════════════════════════════════════════════════════════

GEMM_BODY i8gemm_k_ld1, 1
GEMM_BODY i8gemm_k_ld2, 2
GEMM_BODY i8gemm_k_ld4, 4