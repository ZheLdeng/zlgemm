// i8gemm_k_ldc.S
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
.global _i8gemm_k

#define A_ADDR_ORG  x0
#define B_ADDR      x1
#define C_ADDR_0    x2
#define M           x3
#define K           x4
#define N           x5
#define LDC         x6

#define N_COUNT     x30
#define C_N_BASE    x27
#define M_COUNT     x10
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

#define LDC_B       x9
#define C_M_BASE    x26

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
    mov  x8,  C_M_BASE
    add  x11, x8,  LDC_B
    add  x12, x8,  LDC_B, lsl #1
    add  x14, x8,  LDC_B, lsl #2
    add  x13, x11, LDC_B, lsl #1
    add  x15, x11, LDC_B, lsl #2
    add  x16, x12, LDC_B, lsl #2
    add  x17, x13, LDC_B, lsl #2

    ldp  q0, q1, [x8]
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
// 逆变换：
//   v9=[r0c0,r0c1,r1c0,r1c1], v10=[r0c2,r0c3,r1c2,r1c3]
//   zip1(v9,v10).2d = [v9.d[0],v10.d[0]] = [r0c0|r0c1, r0c2|r0c3] → row0[c0..c3]  ✓
//   zip2(v9,v10).2d = [v9.d[1],v10.d[1]] = [r1c0|r1c1, r1c2|r1c3] → row1[c0..c3]  ✓
.macro STORE_C_ROWMAJOR
    mov  x8,  C_M_BASE
    add  x11, x8,  LDC_B
    add  x12, x8,  LDC_B, lsl #1
    add  x14, x8,  LDC_B, lsl #2
    add  x13, x11, LDC_B, lsl #1
    add  x15, x11, LDC_B, lsl #2
    add  x16, x12, LDC_B, lsl #2
    add  x17, x13, LDC_B, lsl #2

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

    stp  q0, q1, [x8]
    stp  q2, q3, [x11]
    stp  q4, q5, [x12]
    stp  q6, q7, [x13]
    stp  q8, q25, [x14]
    stp  q26, q27, [x15]
    stp  q28, q29, [x16]
    stp  q30, q31, [x17]
.endm

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

_i8gemm_k:
    INIT

    lsl  LDC_B, LDC, #2

    mov  C_N_BASE, C_ADDR_0

    mov  B_ADDR_1, B_ADDR

// ── N 方向外层循环（每次 8 列）─────────────────────────────
401:
    asr  N_COUNT, N, #3
    cmp  N_COUNT, #0
    b.eq 301f

101:
    mov  B_ADDR,  B_ADDR_1
    mov  A_ADDR,  A_ADDR_ORG

    mov  C_M_BASE, C_N_BASE

    asr  M_COUNT, M, #3
    cmp  M_COUNT, #0
    b.eq 301f

// ── M 方向内层循环（每次 8 行）──────────────────────────────
102:
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

    add  C_M_BASE, C_M_BASE, LDC_B, lsl #3

    subs M_COUNT, M_COUNT, #1
    b.gt 102b

    add  C_N_BASE, C_N_BASE, #32

    subs N_COUNT, N_COUNT, #1
    b.gt 101b

301:
    RESTORE
    ret
