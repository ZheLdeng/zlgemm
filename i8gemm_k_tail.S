// i8gemm_k_ld_tail.S
// 尾部 M 内核：i8gemm_k_ld1 / i8gemm_k_ld2 / i8gemm_k_ld4
//
// 接口与 i8gemm_k_ld 完全一致，但 m 参数分别固定为 1 / 2 / 4，
// 不再要求 m 是 8 的倍数——调用方只传剩余的尾部行数。
//
// void i8gemm_k_ldN(const int8_t  *A,
//                   const int8_t  *B_reordered,
//                   int32_t       *C,
//                   int            m,          // 实际行数（对应函数固定为 1/2/4）
//                   int            k,          // K（16 的倍数）
//                   int            n,          // N（8 的倍数）
//                   int8_t        *A_reorder,
//                   int            lda,
//                   int            ldb,        // [sp+0]
//                   int            ldc);       // [sp+8] linux / [sp+4] apple
//
// A_reorder 布局（每 K-block = 8 列，每 block 在 reorder 中的字节数）：
//   m=1 : 16 bytes/block  (row0[8] | zero[8])
//   m=2 : 16 bytes/block  (row0[8] | row1[8])
//   m=4 : 32 bytes/block  (row0[8] | row1[8] | row2[8] | row3[8])
//
// smmla 寄存器组使用：
//   m=1/2 : v9~v12   (1 对，对应 row0/row1 × col0..7)
//   m=4   : v9~v16   (2 对，对应 row0/row1 × col0..7 + row2/row3 × col0..7)

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

// ── INIT / RESTORE ────────────────────────────────────────────────
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