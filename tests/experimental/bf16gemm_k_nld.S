// bf16gemm_k_nld.S — Pure-compute baseline kernel (No-LoaD)
// void bf16gemm_k_nld(const uint16_t *A,            // x0  (unused)
//                     const uint16_t *B_reordered,   // x1  (unused)
//                     float         *C,              // x2  (unused)
//                     int            m,              // x3  M (8 的倍数)
//                     int            k,              // x4  K (8 的倍数)
//                     int            n,              // x5  N (8 的倍数)
//                     uint16_t      *A_reorder,      // x6  (unused)
//                     int            lda,            // x7  (unused)
//                     int            ldb,            // [sp] (unused)
//                     int            ldc);           // [sp] (unused)
//
// 与 bf16gemm_k_ld 保持完全相同:
//   - M-N-K 循环结构
//   - bfmmla 指令数量 (2*(K/8) blocks × 16 bfmmla per block)
//   - 累加器寄存器 v9-v24（16 个，存 8×8 float32 tile）
//
// 去掉:
//   - 所有 ldp/ld1/stp 内存指令
//   - LOAD_C_ROWMAJOR / STORE_C_ROWMAJOR
//   - A_reorder 写入
//   - strided / cached path 分支
//   - 所有地址计算 (add/sub 用于计算 load/store 地址)
//
// 目的: 测量纯 bfmmla 计算吞吐上限，作为 baseline 计算 efficiency

        .text
        .align 4
        .global bf16gemm_k_nld

#define M           x3
#define K           x4
#define N           x5

#define N_COUNT     x30
#define M_COUNT     x10
#define K_COUNT     x23

// ── INIT / RESTORE (same as original) ──────────────────────────
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

// ── ZERO accumulators v9-v24 (replaces LOAD_C_ROWMAJOR) ────────
        .macro ZERO_ACC
        eor v9.16b,  v9.16b,  v9.16b
        eor v10.16b, v10.16b, v10.16b
        eor v11.16b, v11.16b, v11.16b
        eor v12.16b, v12.16b, v12.16b
        eor v13.16b, v13.16b, v13.16b
        eor v14.16b, v14.16b, v14.16b
        eor v15.16b, v15.16b, v15.16b
        eor v16.16b, v16.16b, v16.16b
        eor v17.16b, v17.16b, v17.16b
        eor v18.16b, v18.16b, v18.16b
        eor v19.16b, v19.16b, v19.16b
        eor v20.16b, v20.16b, v20.16b
        eor v21.16b, v21.16b, v21.16b
        eor v22.16b, v22.16b, v22.16b
        eor v23.16b, v23.16b, v23.16b
        eor v24.16b, v24.16b, v24.16b
        .endm

// ── COMPUTE_BLOCK: 16 bfmmla (no loads interleaved) ────────────
// 与原 kernel COMPUTE_A1_B1 中的 bfmmla 序列完全一致
// 只是去掉了前面的 ldp A 和 ldp B
        .macro COMPUTE_BLOCK_A
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

// ── COMPUTE_BLOCK: alternate register set (matching A1×B1 pattern) ──
        .macro COMPUTE_BLOCK_B
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

bf16gemm_k_nld:
        INIT

        // ── M 方向外层循环（每次 8 行）──────────────────────────────
        asr  M_COUNT, M, #3
        cmp  M_COUNT, #0
        b.eq 301f

400:
        // ── N 方向内层循环（每次 8 列）──────────────────────────────
        asr  N_COUNT, N, #3
        cmp  N_COUNT, #0
        b.eq 301f

102:
        // ── Zero accumulators（替换 LOAD_C_ROWMAJOR）────────────────
        ZERO_ACC

        // ── K 内层循环 ──────────────────────────────────────────────
        // 与原 kernel 完全相同的迭代次数和 bfmmla 数量:
        //   K_COUNT = K/8
        //   每个迭代: 2 个 compute block (COMPUTE_BLOCK_A + COMPUTE_BLOCK_B)
        //   总计: 2*(K/8) blocks × 16 bfmmla = 4*K bfmmla per tile
        asr  K_COUNT, K, #3
        cmp  K_COUNT, #0
        b.eq 301f

        subs K_COUNT, K_COUNT, #1
        b.eq 202f                   // K=8 → 直接进 tail

        // 主循环: 每次 2 个 block × 16 bfmmla
201:
        COMPUTE_BLOCK_A
        COMPUTE_BLOCK_B
        subs K_COUNT, K_COUNT, #1
        b.gt 201b

        // Tail: 最后一批 2 个 block
202:
        COMPUTE_BLOCK_A
        COMPUTE_BLOCK_B

        // ── Next N column ──
        subs N_COUNT, N_COUNT, #1
        b.gt 102b

        // ── Next M row ──
        subs M_COUNT, M_COUNT, #1
        b.gt 400b

301:
        RESTORE
        ret
