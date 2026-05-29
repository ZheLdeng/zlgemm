// i8gemm_k_ldc.S
// 版本1：在 .S 内直接输出行主序 C，消除外部 reorder_C_2x2 调用
//
// void i8gemm_k(const int8_t  *A_reordered,   // x0
//               const uint8_t *B_reordered,   // x1
//               int32_t       *C,             // x2  行主序输出，stride = ldc
//               int            m,             // x3
//               int            k,             // x4
//               int            n,             // x5
//               int            ldc);          // x6  C 每行元素数
//
// 内核每次处理 8×8 输出块，v9~v24 的 smmla 格式：
//   v9  = [C[r0][c0], C[r0][c1], C[r1][c0], C[r1][c1]]
//   v10 = [C[r0][c2], C[r0][c3], C[r1][c2], C[r1][c3]]
//   v11 = [C[r0][c4], C[r0][c5], C[r1][c4], C[r1][c5]]
//   v12 = [C[r0][c6], C[r0][c7], C[r1][c6], C[r1][c7]]
//   v13~v16: r2/r3 同上, v17~v20: r4/r5, v21~v24: r6/r7

.text
.align 4
.global i8gemm_k

// ── 寄存器别名（M_COUNT 改为 x10，LDC 占用原 M_COUNT 的 x6 位置）
#define A_ADDR_ORG  x0
#define B_ADDR      x1
#define C_ADDR_0    x2
#define M           x3
#define K           x4
#define N           x5
#define LDC         x6      // 新增参数（原 M_COUNT=x6 移到 x10）

#define N_COUNT     x30
#define C_N_BASE    x27     // 当前 N 块的列起点地址（原 C_ADDR_1，但语义不同）
#define M_COUNT     x10     // 原 x6，按需求改为 x10
#define C_ADDR_3    x11
#define C_ADDR_4    x25
#define B_ADDR_1    x19
#define C_UNROLL    x24
#define C_OFFSET    x20
#define B_ADDR_3    x21
#define OFFSET      x22
#define K_COUNT     x23
#define K_TAIL      x28
#define A_ADDR      x29

#define LDC_B       x9      // ldc * 4（字节步长，预计算）
#define C_M_BASE    x26     // 当前 M 块 row0 的行主序地址

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
// 原理：行主序 row_i: [ci_c0,ci_c1,ci_c2,...,ci_c7]（两个 q）
// blocked 格式需要：
//   v9  = [r0c0,r0c1, r1c0,r1c1] = [row0.d[0], row1.d[0]] → zip1(row0_lo,row1_lo).2d
//   v10 = [r0c2,r0c3, r1c2,r1c3] = [row0.d[1], row1.d[1]] → zip2(row0_lo,row1_lo).2d
//   v11 = [r0c4,r0c5, r1c4,r1c5] = [row0_hi.d[0], row1_hi.d[0]] → zip1(row0_hi,row1_hi).2d
//   v12 = [r0c6,r0c7, r1c6,r1c7] = [row0_hi.d[1], row1_hi.d[1]] → zip2(row0_hi,row1_hi).2d
//
// 使用临时寄存器 x7,x8,x11~x17（caller-saved，无需保存）
.macro LOAD_C_ROWMAJOR
    // 8 行地址（C_M_BASE 已指向 row0）
    mov  x8,  C_M_BASE
    add  x11, x8,  LDC_B
    add  x12, x11, LDC_B
    add  x13, x12, LDC_B
    add  x14, x13, LDC_B
    add  x15, x14, LDC_B
    add  x16, x15, LDC_B
    add  x17, x16, LDC_B

    // row0(v0_lo,v0_hi), row1(v1_lo,v1_hi) → v9~v12
    ldp  q0, q1, [x8]           // row0: [c0~c3]=v0, [c4~c7]=v1
    ldp  q2, q3, [x11]          // row1: [c0~c3]=v2, [c4~c7]=v3
    zip1 v9.2d,  v0.2d, v2.2d   // [r0c0,r0c1, r1c0,r1c1]
    zip2 v10.2d, v0.2d, v2.2d   // [r0c2,r0c3, r1c2,r1c3]
    zip1 v11.2d, v1.2d, v3.2d   // [r0c4,r0c5, r1c4,r1c5]
    zip2 v12.2d, v1.2d, v3.2d   // [r0c6,r0c7, r1c6,r1c7]

    // row2, row3 → v13~v16
    ldp  q0, q1, [x12]
    ldp  q2, q3, [x13]
    zip1 v13.2d, v0.2d, v2.2d
    zip2 v14.2d, v0.2d, v2.2d
    zip1 v15.2d, v1.2d, v3.2d
    zip2 v16.2d, v1.2d, v3.2d

    // row4, row5 → v17~v20
    ldp  q0, q1, [x14]
    ldp  q2, q3, [x15]
    zip1 v17.2d, v0.2d, v2.2d
    zip2 v18.2d, v0.2d, v2.2d
    zip1 v19.2d, v1.2d, v3.2d
    zip2 v20.2d, v1.2d, v3.2d

    // row6, row7 → v21~v24
    ldp  q0, q1, [x16]
    ldp  q2, q3, [x17]
    zip1 v21.2d, v0.2d, v2.2d
    zip2 v22.2d, v0.2d, v2.2d
    zip1 v23.2d, v1.2d, v3.2d
    zip2 v24.2d, v1.2d, v3.2d
.endm

// ── STORE_C_ROWMAJOR：将 blocked v9~v24 按行主序写回 C ────────
// 逆变换：
//   v9=[r0c0,r0c1,r1c0,r1c1], v10=[r0c2,r0c3,r1c2,r1c3]
//   zip1(v9,v10).2d = [v9.d[0],v10.d[0]] = [r0c0|r0c1, r0c2|r0c3] → row0[c0..c3]  ✓
//   zip2(v9,v10).2d = [v9.d[1],v10.d[1]] = [r1c0|r1c1, r1c2|r1c3] → row1[c0..c3]  ✓
.macro STORE_C_ROWMAJOR
    // 8 行地址（C_M_BASE 指向当前块 row0）
    mov  x8,  C_M_BASE
    add  x11, x8,  LDC_B
    add  x12, x11, LDC_B
    add  x13, x12, LDC_B
    add  x14, x13, LDC_B
    add  x15, x14, LDC_B
    add  x16, x15, LDC_B
    add  x17, x16, LDC_B

    // row0, row1 ← v9~v12
    zip1 v0.2d, v9.2d,  v10.2d  // row0[c0..c3]
    zip1 v1.2d, v11.2d, v12.2d  // row0[c4..c7]
    zip2 v2.2d, v9.2d,  v10.2d  // row1[c0..c3]
    zip2 v3.2d, v11.2d, v12.2d  // row1[c4..c7]
    stp  q0, q1, [x8]
    stp  q2, q3, [x11]

    // row2, row3 ← v13~v16
    zip1 v0.2d, v13.2d, v14.2d
    zip1 v1.2d, v15.2d, v16.2d
    zip2 v2.2d, v13.2d, v14.2d
    zip2 v3.2d, v15.2d, v16.2d
    stp  q0, q1, [x12]
    stp  q2, q3, [x13]

    // row4, row5 ← v17~v20
    zip1 v0.2d, v17.2d, v18.2d
    zip1 v1.2d, v19.2d, v20.2d
    zip2 v2.2d, v17.2d, v18.2d
    zip2 v3.2d, v19.2d, v20.2d
    stp  q0, q1, [x14]
    stp  q2, q3, [x15]

    // row6, row7 ← v21~v24
    zip1 v0.2d, v21.2d, v22.2d
    zip1 v1.2d, v23.2d, v24.2d
    zip2 v2.2d, v21.2d, v22.2d
    zip2 v3.2d, v23.2d, v24.2d
    stp  q0, q1, [x16]
    stp  q2, q3, [x17]
.endm

// ── 计算宏（与原版完全相同）─────────────────────────────────────
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

// ── 函数体 ───────────────────────────────────────────────────
// 循环结构与原版完全对应，仅 C 指针管理方式改变：
//   原版：C_ADDR_0/C_ADDR_1 顺序读写 blocked 格式（指针自动推进）
//   新版：C_M_BASE 跟踪当前 8×8 块在行主序 C 中的 row0 地址
//         C_N_BASE（x27）跟踪 N 循环的列起点（每次 N 迭代 +32 字节）
//         每次 M 迭代 C_M_BASE += 8 * LDC_B
//
// B 指针推进逻辑与原版完全相同（B_ADDR / B_ADDR_1 交换机制不变）
i8gemm_k:
    INIT

    lsl  LDC_B, LDC, #2         // LDC_B(x9) = ldc * 4 字节

    // C_N_BASE 初始指向 C[0][0]，N 循环每次 +32（8列 * 4字节）
    mov  C_N_BASE, C_ADDR_0     // x27 = C 基址

    // B_ADDR_1 保存整个计算的 B 起点（原版逻辑）
    mov  B_ADDR_1, B_ADDR

// ── N 方向外层循环（每次 8 列）─────────────────────────────
401:
    asr  N_COUNT, N, #3
    cmp  N_COUNT, #0
    b.eq 301f

101:
    // 原版：mov B_ADDR, B_ADDR_1（N 循环用 B_ADDR 保存 N-tile 起点）
    mov  B_ADDR,  B_ADDR_1
    mov  A_ADDR,  A_ADDR_ORG

    // C_M_BASE 从当前列块的第 0 行开始
    mov  C_M_BASE, C_N_BASE

    asr  M_COUNT, M, #3
    cmp  M_COUNT, #0
    b.eq 301f

// ── M 方向内层循环（每次 8 行）──────────────────────────────
102:
    // 原版：mov B_ADDR_1, B_ADDR（保存当前 B 块起点供 K 循环使用）
    mov  B_ADDR_1, B_ADDR

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

    STORE_C_ROWMAJOR

    // C_M_BASE 下移 8 行
    add  C_M_BASE, C_M_BASE, LDC_B, lsl #3   // += 8 * LDC_B

    subs M_COUNT, M_COUNT, #1
    b.gt 102b

    // N 循环结束：C_N_BASE 右移 8 列，B_ADDR_1 已由 K 循环推进到下一个 N-tile
    add  C_N_BASE, C_N_BASE, #32

    subs N_COUNT, N_COUNT, #1
    b.gt 101b

301:
    RESTORE
    ret
