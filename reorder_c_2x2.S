// reorder_c_2x2.S
// NEON 加速的 reorder_C_2x2
//
// void reorder_c_2x2_neon(const int32_t *C_blocked,   // x0
//                          int32_t       *C_normal,    // x1
//                          int            M,           // x2
//                          int            N);          // x3
//
// 功能：将 2×2 blocked 格式的 C 还原为普通行主序
//
// 输入 C_blocked 内存布局（C 代码 reorder_C_2x2 的逆操作）：
//   顺序为 block(rb=0,cb=0), block(0,1), ..., block(0,N/2-1),
//          block(1,0), block(1,1), ...
//   每个 block 4 个 int32：[row_even_col_a, row_even_col_b, row_odd_col_a, row_odd_col_b]
//
// NEON 策略（每次处理 4 个连续 block = 8 列宽，对应 2 行输出）：
//   加载 4 个 2×2 block（v0~v3，每个 = 1 个 block）
//   uzp1 on .2d → 提取偶数行 8 个 int32 → 写入 row_even
//   uzp2 on .2d → 提取奇数行 8 个 int32 → 写入 row_odd
//
// 对 N%8 != 0 的尾部（2 个 block = 4 列）：
//   加载 v0, v1，uzp1/uzp2，写 2×4 int32（各 16 字节）

.text
.align 4
.global reorder_c_2x2_neon

// 寄存器别名
#define C_BLK    x0     // blocked 输入指针（顺序推进）
#define C_NRM    x1     // 行主序输出基址
#define M_REG    x2
#define N_REG    x3
#define RB       x4     // 行块循环变量（递减）
#define CB8      x5     // 每行的 8 列列块数 = N/8
#define CB2      x6     // 每行的 2 列尾块数 = (N%8)/2 = (N/2)%4
#define ROW_EVEN x7     // 当前偶数行在 C_normal 的起始地址
#define ROW_ODD  x8     // 当前奇数行 = ROW_EVEN + N_BYTES
#define N_BYTES  x9     // N * 4
#define CB_TMP   x10    // cb 循环内临时计数
#define ROW_TMP_E x11   // cb 循环内列偏移写指针（偶数行）
#define ROW_TMP_O x12   // cb 循环内列偏移写指针（奇数行）

reorder_c_2x2_neon:
    stp x29, x30, [sp, #-16]!
    stp x19, x20, [sp, #-16]!

    // 预计算
    lsl  N_BYTES, N_REG, #2     // N_BYTES = N * 4
    lsr  CB8, N_REG, #3         // CB8 = N / 8（每次处理 8 列 = 4 个 2×2 block）
    lsr  x13, N_REG, #1         // N/2
    and  CB2, x13, #3           // CB2 = (N/2) % 4（每次处理 2 列的尾块数）
    lsr  RB, M_REG, #1          // RB = M / 2（行块数）

    // ROW_EVEN 初始指向 C_normal 第 0 行
    mov  ROW_EVEN, C_NRM
    add  ROW_ODD,  ROW_EVEN, N_BYTES

// ── 外层循环：行块（每次 2 行）────────────────────────────────
rb_loop:
    cbz  RB, done

    // 在列循环内用临时指针，结束后 ROW_EVEN/ROW_ODD 不变（整行写完后再推进）
    mov  ROW_TMP_E, ROW_EVEN
    mov  ROW_TMP_O, ROW_ODD

    // ── 8 列主循环（每次 4 个 2×2 block）──────────────────────
    mov  CB_TMP, CB8
    cbz  CB_TMP, tail_4col

cb8_loop:
    // 加载 4 个连续 2×2 block（64 字节）
    // v0 = [r_even_c0, r_even_c1, r_odd_c0, r_odd_c1]  (block 0)
    // v1 = [r_even_c2, r_even_c3, r_odd_c2, r_odd_c3]  (block 1)
    // v2 = [r_even_c4, r_even_c5, r_odd_c4, r_odd_c5]  (block 2)
    // v3 = [r_even_c6, r_even_c7, r_odd_c6, r_odd_c7]  (block 3)
    ldp  q0, q1, [C_BLK], #32
    ldp  q2, q3, [C_BLK], #32

    // uzp1 on .2d：取 .d[0]（偶数行的两个元素）
    // uzp1(v0,v1).2d = [v0.d[0], v1.d[0]] = [r_even_c0|c1, r_even_c2|c3]
    uzp1 v4.2d, v0.2d, v1.2d   // row_even: c0..c3
    uzp1 v5.2d, v2.2d, v3.2d   // row_even: c4..c7
    // uzp2 on .2d：取 .d[1]（奇数行的两个元素）
    uzp2 v6.2d, v0.2d, v1.2d   // row_odd:  c0..c3
    uzp2 v7.2d, v2.2d, v3.2d   // row_odd:  c4..c7

    stp  q4, q5, [ROW_TMP_E], #32
    stp  q6, q7, [ROW_TMP_O], #32

    subs CB_TMP, CB_TMP, #1
    b.gt cb8_loop

    // ── 4 列尾块（每次 2 个 2×2 block，当 N%8==4 时执行）──
tail_4col:
    // CB2 可能为 0, 1, 2, 3，每次处理 2 个 2×2 block（4 列）
    mov  CB_TMP, CB2
    cbz  CB_TMP, row_done

cb2_loop:
    // 加载 2 个连续 2×2 block（32 字节）
    ldp  q0, q1, [C_BLK], #32
    // v0 = [r_even_c0,r_even_c1, r_odd_c0,r_odd_c1]
    // v1 = [r_even_c2,r_even_c3, r_odd_c2,r_odd_c3]

    uzp1 v2.2d, v0.2d, v1.2d   // row_even: c0..c3
    uzp2 v3.2d, v0.2d, v1.2d   // row_odd:  c0..c3

    str  q2, [ROW_TMP_E], #16
    str  q3, [ROW_TMP_O], #16

    subs CB_TMP, CB_TMP, #1
    b.gt cb2_loop

row_done:
    // 行指针推进 2 行
    add  ROW_EVEN, ROW_EVEN, N_BYTES, lsl #1   // += 2 * N_BYTES
    add  ROW_ODD,  ROW_ODD,  N_BYTES, lsl #1

    subs RB, RB, #1
    b.gt rb_loop

done:
    ldp x19, x20, [sp], #16
    ldp x29, x30, [sp], #16
    ret
