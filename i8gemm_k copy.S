// sme_matrix.S
.text
.align 4
//gcc -g -march=armv8.2-a+i8mm -o calculate calculatei8mm.c i8gemm_k1.S
// void _i8gemm_k( *A, *B, *C, int m, int K, int n)
.global i8gemm_k 

#define A_ADDR_ORG x0
#define B_ADDR     x1
#define C_ADDR_0   x2
#define M          x3
#define K          x4
#define N          x5
#define N_COUNT    x30
#define C_ADDR_1   x27
#define C_ADDR_2   x10
#define C_ADDR_3   x11
#define C_ADDR_4   x25
#define B_ADDR_1   x19
#define C_UNROLL   x24
#define C_OFFSET   x20
#define B_ADDR_3   x21
#define OFFSET     x22         
#define K_COUNT    x23
#define M_COUNT    x6
#define K_TAIL     x28
#define A_ADDR     x29

.macro INIT
    // 保存被调用者保存寄存器
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

.macro LOAD_C

    ldp q9, q10, [C_ADDR_0], #32
    ldp q11, q12, [C_ADDR_0], #32

    ldp q13, q14, [C_ADDR_0], #32
    ldp q15, q16, [C_ADDR_0], #32

    ldp q17, q18, [C_ADDR_0], #32
    ldp q19, q20, [C_ADDR_0], #32

    ldp q21, q22, [C_ADDR_0], #32
    ldp q23, q24, [C_ADDR_0], #32

.endm

.macro STORE_C

    stp q9, q10, [C_ADDR_1], #32
    stp q11, q12, [C_ADDR_1], #32

    stp q13, q14, [C_ADDR_1], #32
    stp q15, q16, [C_ADDR_1], #32

    stp q17, q18, [C_ADDR_1], #32
    stp q19, q20, [C_ADDR_1], #32

    stp q21, q22, [C_ADDR_1], #32
    stp q23, q24, [C_ADDR_1], #32

.endm
.macro LOAD_A0_B0
    ldp q0, q1,[A_ADDR], #32
    ldp q2, q3,[A_ADDR], #32

    ldp q4, q5,[B_ADDR_1], #32
    ldp q6, q7,[B_ADDR_1], #32
.endm

.macro COMPUTE_A1_B1

    ldp q8, q25,[A_ADDR], #32
    ldp q26, q27,[A_ADDR], #32

    ldp q28, q29,[B_ADDR_1], #32
    ldp q30, q31,[B_ADDR_1], #32

    smmla v9.4s, v0.16b, v4.16b
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

    ldp q0, q1,[A_ADDR], #32
    ldp q2, q3,[A_ADDR], #32

    ldp q4, q5,[B_ADDR_1], #32
    ldp q6, q7,[B_ADDR_1], #32

    smmla v9.4s, v8.16b, v28.16b
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

i8gemm_k:
    INIT
    mov C_ADDR_1, C_ADDR_0
    mov B_ADDR_1, B_ADDR
401:
    asr N_COUNT, N, #3
    cmp N_COUNT, #0
    b.eq 301f
101:
    mov B_ADDR, B_ADDR_1
    mov A_ADDR, A_ADDR_ORG
    asr M_COUNT, M, #3
    cmp M_COUNT, #0
    b.eq 301f
102:
    mov B_ADDR_1, B_ADDR
    LOAD_C
    asr K_COUNT, K, #4           // 内层循环计数器 k_depth
    cmp K_COUNT, #0
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
    STORE_C
    subs M_COUNT, M_COUNT, #1
    b.gt 102b
    subs N_COUNT, N_COUNT, #1
    b.gt 101b
301:
    // 恢复寄存器
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
    ret