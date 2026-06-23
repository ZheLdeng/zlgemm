# i8gemm 对照 KleidiAI 的改动与差异说明

本文档补充说明本轮在 i8gemm 中参考 KleidiAI I8 NEON I8MM 路径做了哪些改动，以及当前实现和 KleidiAI 仍然有哪些关键差异。

## 本轮改动

### 1. 新增 16x4 I8MM 实验路径

新增文件：

```text
lib/i8gemm_m16n4.c
lib/i8gemm_m16n4.S
```

新增入口：

```c
void i8gemm_mt_dispatch_m16n4(const i8_t *A, const i8_t *B_reo,
                              i32_t *C, int M, int K_r, int N_r,
                              int nthreads);
```

该路径保持 i8gemm 现有数据契约：

| Tensor | 当前格式 |
|---|---|
| A | row-major int8, `M x K_r` |
| B | 由 `i8_pack_B` 生成的 packed B |
| C | row-major int32, `M x N_r` |

计算块形状参考 KleidiAI I8MM ukernel：`M=16, N=4, K step=8`。每个 `smmla` accumulator 对应 2 行 x 2 列的 int32 输出。完整 `16x4` 主块已经转为 `lib/i8gemm_m16n4.S` 实现；zero-C 下的 M tail 也已经走 `.S` 条件 store。禁用 packed-A、禁用 asm、或 accumulate-C 下的 M tail 继续走 C/ACLE 回退路径。

### 2. 16x4 增加 packed-A 复用

上一版 16x4 的主要问题是：每个 N4 panel 都从 row-major A 重新跨行加载，导致宽 N 场景重复读取 A。

本轮增加了 `i8_pack_A_m16n4`：

```c
static void i8_pack_A_m16n4(const i8_t *A, i8_t *P, int M, int K_r,
                            int num_threads);
```

pack 后每个 M16 block 的布局为：

```text
for kb in K step 8:
  for rp in 0..7:
    pack rows (2*rp, 2*rp+1), 8 bytes each
```

也就是每个 K8 slice 存 8 个 `int8x16_t`，共 `16 rows x 8 K = 128 bytes`。后续所有 N4 panel 都复用同一份 packed A。

可通过环境变量回退旧路径：

```bash
I8_M16N4_PACK_A=0
```

也可以只禁用 16x4 asm 主块，保留 packed-A，但回退到 C/ACLE microkernel：

```bash
I8_M16N4_USE_ASM=0
```

### 3. 16x4 增加 zero-C fast path

测试和 i8gemm 头文件都约定 int32 C 在调用前为 0。16x4 现在默认不读 C，accumulator 从 0 初始化。

如果需要旧的 accumulate-C 语义，可设置：

```bash
I8_GEMM_ACCUMULATE_C=1
```

### 4. 8x8 NEON 增加 zero-C asm kernel

修改文件：

| 文件 | 改动 |
|---|---|
| `lib/i8gemm_k.S` | 增加 `LOAD_ZERO_*` 和 `i8gemm_k_zero*` / `i8gemm_k_reo_zero*` |
| `lib/i8gemm_mt.c` | 默认 int32 dispatch 调用 zero-C kernel |

新增符号包括：

```text
i8gemm_k_zero
i8gemm_k_zero1
i8gemm_k_zero2
i8gemm_k_zero4
i8gemm_k_reo_zero
i8gemm_k_reo_zero1
i8gemm_k_reo_zero2
i8gemm_k_reo_zero4
```

这些路径跳过 C load，直接 `movi accumulator, #0`，然后计算并写回 int32 row-major C。

同样可以用 `I8_GEMM_ACCUMULATE_C=1` 回退旧的读 C 累加路径。

### 5. 16x4 tail 与指令排布继续下沉到 `.S`

修改文件：

| 文件 | 改动 |
|---|---|
| `lib/i8gemm_m16n4.S` | 增加 `rows` 参数，store 阶段按有效 row-pair 条件写回 |
| `lib/i8gemm_m16n4.c` | zero-C 时允许 `rows < 16` 的 M tail 调用 asm |

当前 `.S` 函数接口：

```c
void i8gemm_k_m16n4_packed_asm(const i8_t *A_pack, const i8_t *B_reo,
                               i32_t *C, int K_r, int N_r, int n0,
                               int rows, int zero_c);
```

tail store 仍保持 int32 row-major 输出。每个 accumulator pair 先通过 `zip1/zip2` 从 blocked 形态恢复成两行连续 4 列，然后根据 `rows` 只写有效行：

```asm
zip1 v0.2d, acc0.2d, acc1.2d
zip2 v1.2d, acc0.2d, acc1.2d
str  q0, [row0]
str  q1, [row1]   // 仅 row1 有效时执行
```

这和 i8gemm 8x8 tail 的思路一致：不要改变对外 C 的 row-major 语义，只是在 store 端做边界保护。与 KleidiAI 相比，当前实现只补了 M tail；N tail 在现有测试里由 `N_r` padded 到 8 且 N4 block 完整覆盖，因此还没有做 partial-column `.S` store。

指令排布也做了收敛：16x4 asm 主循环现在按 K32 展开，每组覆盖 4 个 K8 slice。第一个 slice 先完整 load，随后在当前 slice 的 `smmla` 计算窗口里预装下一 slice 的 B 和 A。这样不再是单纯的 K8 `load phase -> compute phase`，而是更接近 KleidiAI 的跨 slice 双缓冲。

```asm
// slice0 current
load B0, A0

// compute slice0 while preloading slice1
load B1, A1[0..3]
smmla slice0 row-pair 0
load A1[4]
smmla slice0 row-pair 1
load A1[5]
...

// repeat for slice1 -> slice2, slice2 -> slice3
// compute slice3, then loop to next K32 group
```

实现上仍保留 16 个 accumulator 在 `v16..v31`。临时寄存器 `v0..v15` 分成 current slice 和 next slice 两套轮换：B 使用 2 个寄存器，A 使用 8 个寄存器。由于寄存器已经很紧，当前版本没有跨 K32 group 预取下一组的第一个 slice；每个 K32 group 内部做 3 次 next-slice 预装，K32 余数走 K8 tail loop。

这比上一版 K8 分段式调度更接近 KleidiAI 的结构，但当前版本需要在函数入口保存/恢复 `d8..d15` 以满足 AArch64 ABI。修正 ABI 后，K32 展开并没有形成稳定净收益，说明当前指令间隔和寄存器选择还没有调到位。它仍比 KleidiAI 保守：没有配合 N4 专用 RHS pack，也没有针对特定核做更细的 `.inst` 级指令间隔调参。

### 6. benchmark 接入

修改文件：

| 文件 | 改动 |
|---|---|
| `tests/bench_dispatch_types.c` | 支持 `i8gemm_neon_m16n4` impl 名称 |
| `tests/run_three_lib_grid.py` | 将 `i8gemm_neon_m16n4` 和 `i8gemm_m16n4.S` 加入 I8 测试构建 |
| `tests/test_i8_m16n4_correctness.c` | 新增 16x4 正确性测试 |

完整复测输出：

```text
results/m8/three_lib_grid_m16n4_opt_full_reps1.csv
```

结果：`17280` 行全部 `ok`。

## 与 KleidiAI 的主要差异

### 1. 输出类型不同

| 项目 | i8gemm 16x4 | KleidiAI I8 16x4 |
|---|---|---|
| accumulator | int32 | int32 |
| 输出 | int32 row-major | f32 row-major |
| 后处理 | 无 dequant/clamp | dequant + clamp |

KleidiAI 的 `kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm` 会把 int32 accumulator 转换成 f32，并应用 scale/clamp。i8gemm 当前目标是 int32 原始输出，所以没有复用 KleidiAI 的完整 store/dequant 路径。

### 2. LHS/A pack 格式不同

KleidiAI 的 LHS pack 是 `qai8dxp4x8`，含义大致是：

```text
mr = 4
kr = 8
k rounded to 32
每 4 行一组 packed int8
每行还有 multiplier / zero-point 元数据
16x4 ukernel 一次使用 4 个 mr=4 子块组成 16 行
```

i8gemm 当前 16x4 packed-A 更简单：

```text
M16 block
K 每 8 一组
每组保存 8 个 row-pair int8x16
只保存 int8 数据
无 multiplier / zero-point 元数据
```

原因是 i8gemm 输出 int32，不需要量化输出元数据。这也意味着当前 pack 格式不能直接喂给 KleidiAI 原 ukernel。

### 3. RHS/B pack 格式仍未 N4 专用化

KleidiAI RHS 是 `qsi8cxp4x8`，专门匹配 `N=4, K step=8` 的 ukernel，且包含 scale、bias、rhs sum 等后处理所需信息。

i8gemm 当前仍复用原有 8 列 B pack：

```text
for each N-block (8 cols):
  for each K-block (8 rows):
    for each col in 8:
      for each row in 8:
        B[row][col]
```

16x4 kernel 只取其中 4 列，相当于在 8 列 panel 内取半块。这样能兼容现有 API 和 benchmark，但不是 N4 最优格式。当前 16x4 和 KleidiAI 的最大结构性差距之一就是这里。

### 4. microkernel 实现方式不同

| 项目 | i8gemm 16x4 | KleidiAI 16x4 |
|---|---|---|
| 完整 16x4 主块 | 独立 `.S` | 手写 inline asm |
| tail / 回退 | zero-C M tail 已在 `.S`；其他回退 C/ACLE `vmmlaq_s32` | 专门 tail path |
| 指令 | 汇编主块显式 `smmla` | 显式 `.inst smmla` |
| load/compute 调度 | K32 内跨 slice 双缓冲 | 手工深度交错 |
| K loop 展开 | K32 主循环，K8 tail loop | 手工展开，按 32 K block |

i8gemm 16x4 主块现在已经不是纯 C/ACLE；`i8gemm_k_m16n4_packed_asm` 会显式维护 16 个 accumulator，并对每个 K8 slice 做 8 次 A load、2 次 B load、16 条 `smmla`。当前版本按 K32 展开：slice0 计算时预装 slice1，slice1 计算时预装 slice2，slice2 计算时预装 slice3。结构上这减少了 load-use stall 和 loop branch 频率，但寄存器压力很高，需要保存/恢复 callee-saved SIMD 寄存器；实测它仍没有超过默认 8x8 和 KleidiAI。它也没有 KleidiAI 那种完整库级 packing/workspace/scheduler 配合，因此目前只能视为实验路径。

### 5. C/store 路径不同

i8gemm 当前完整 16x4 asm 主块会把 blocked accumulator 通过 `zip1/zip2` 转成 row-major，然后每两行使用两个 `str q` 写回 int32：

```asm
zip1 v0.2d, acc0.2d, acc1.2d
zip2 v1.2d, acc0.2d, acc1.2d
str  q0, [row0]
str  q1, [row1]
```

zero-C M tail 现在也走 asm 条件 store；accumulate-C M tail、禁用 asm 或禁用 packed-A 时仍走 C/ACLE 回退路径，那里会逐 lane 写 int32。accumulate-C 的 tail 没有贸然走 asm，是因为当前 C load 路径会加载完整 16 行，`rows < 16` 时可能越界读用户 C。KleidiAI 的完整块会按 row-major f32 store，并对尾列做专门处理；两者的输出类型和 tail 策略仍不相同。

### 6. 调度策略不同

i8gemm 16x4 当前使用简单的 2D work unit：

```text
work = M16 block x N4 block
OpenMP static schedule
```

KleidiAI ukernel 本身只负责 packed microkernel，实际库级性能依赖调用方如何做 pack、workspace、M/N split。当前 benchmark adapter 是按 M blocks 做 OpenMP 静态切分；ACL 则有更完整的 shape selector 和 workspace scheduler。

i8gemm 当前还没有根据 shape 自动选择：

```text
8x8 vs 16x4
M split vs N split vs 2D split
是否 K split
是否使用 N4 专用 B pack
```

因此 16x4 目前仍作为显式 benchmark 路径 `i8gemm_neon_m16n4`，还没有接入默认 selector。

## 性能变化摘要

完整复测文件：

```text
results/m8/three_lib_grid_m16n4_opt_full_reps1.csv
```

GT_L2 中位数：

| Threads | i8gemm NEON 8x8 | i8gemm NEON m16n4 | KleidiAI I8 NEON I8MM |
|---:|---:|---:|---:|
| 1 | 343.8 | 245.7 | 364.4 |
| 2 | 545.4 | 402.7 | 544.8 |
| 4 | 832.0 | 666.2 | 862.5 |
| 8 | 985.2 | 885.2 | 1040.8 |

16x4 相比上一版的 GT_L2 中位数提升：

```text
1c: 170.0 -> 245.7
2c: 311.5 -> 402.7
4c: 537.1 -> 666.2
8c: 789.7 -> 885.2
```

代表点 `2048x4096x8192, 8c`：

```text
i8gemm NEON 8x8      1909.8 GOPS
i8gemm NEON m16n4    1496.8 GOPS
KleidiAI I8MM        1793.3 GOPS
```

补充 quick 复测文件：

```text
results/m8/three_lib_grid_m16n4_asm_quick.csv
results/m8/three_lib_grid_m16n4_asm_tail_quick.csv
```

本轮 K32 双缓冲并修正 `d8..d15` ABI 保存后的手动 spot check：

```text
M=64, K=512, N=512, 1c:    261.55 GOPS
M=64, K=512, N=512, 8c:    142.23 GOPS
M=512, K=1024, N=1024, 1c: 300.68 GOPS
M=512, K=1024, N=1024, 8c: 586.09 GOPS
M=2048, K=4096, N=8192, 1c: 225.70 GOPS
M=2048, K=4096, N=8192, 8c: 1592.91 GOPS
```

验证项：

```text
m16n4 correctness ok
I8_GEMM_ACCUMULATE_C=1 m16n4 correctness ok
objdump: lib/i8gemm_m16n4.S 对象文件包含 smmla
```

上一轮 `2048x4096x8192, 8c` quick 对照：

```text
i8gemm NEON 8x8      1922.6 GOPS
i8gemm NEON m16n4    1582.1 GOPS
KleidiAI I8MM        1805.0 GOPS
```

结论：packed-A、zero-C、`.S` 主块和 zero-C M tail 已经改善 16x4 的结构；K32 双缓冲当前是实验性改动，修正 ABI 后还没有稳定超过默认 8x8 或 KleidiAI。要继续逼近 KleidiAI，还需要补 N4 专用 B pack，并继续优化 16x4 asm 的更细粒度 load/compute 间隔和 store path。

## 下一步建议

1. 为 16x4 增加 N4 专用 B pack，避免从 8 列 `B_reo` 中取半块。
2. 继续优化 16x4 asm：在当前 K32 框架内做更细粒度指令间隔调参，并评估是否跨 K32 group 预取下一组 slice0。
3. 如果未来允许任意未 padding 的 N，补 partial-column `.S` store；当前测试里 N 已 padding 到 N4/N8。
4. 做 shape selector：窄 N、小 M tail、多线程 work unit 不足时尝试 16x4；宽 N 大矩阵优先保留 8x8。
5. 如果未来要对齐 KleidiAI 的 f32 输出口径，则需要单独增加 dequant/clamp 输出路径，而不是复用当前 int32 C API。

## 对照 ACL 的新增调度优化

ACL I8 的 `gemm_int8.cpp` 会在 `sve_hybrid_s8s32_mmla_6x4VL`、`sve_interleaved_s8s32_mmla_8x3VL`、`a64_interleaved_s8s32_mmla_8x12`、`a64_hybrid_s8s32_mmla_6x16` 等 kernel family 之间做估算选择。这个 selector 的通用思想是：中小 M、medium panel 和大矩阵不能只用同一套 M split。

对照 ACL 后，本轮先在 i8gemm SVE I8 dispatch 加了一个保守的 N-split 规则：

```c
if (M <= 512 && K_r >= 512 && N_r >= 512)
    return 1;
```

这个规则只在 `n_tiles >= num_threads` 且多线程路径里生效，目的是让 `K,N >= 512` 的中等 shape 更早使用 N-split，复用 A reorder 并改善 4/8 核 work distribution。大 M 窄 N 仍保留原有 M-split 优先策略。

验证：

```text
SVE correctness OK: 6000 wrapper+dispatch cases
```

代表点 `K=N=512` 的强制 split 探针显示 N-split 明显更适合中小 M：

```text
M=64,  8c: M-split 1020.03 GOPS, N-split 1985.86 GOPS
M=128, 8c: M-split 1103.62 GOPS, N-split 2468.63 GOPS
M=256, 8c: M-split 2042.64 GOPS, N-split 2528.81 GOPS
M=512, 8c: M-split 2211.33 GOPS, N-split 2607.51 GOPS
```

应用自动规则后的 spot check：

```text
M=64,   K=512,  N=512,  1c: 466.82 GOPS
M=64,   K=512,  N=512,  8c: 1834.09 GOPS
M=128,  K=512,  N=512,  1c: 484.57 GOPS
M=128,  K=512,  N=512,  8c: 2239.02 GOPS
M=512,  K=512,  N=512,  1c: 486.33 GOPS
M=512,  K=512,  N=512,  8c: 2493.05 GOPS
M=512,  K=1024, N=1024, 1c: 536.26 GOPS
M=512,  K=1024, N=1024, 8c: 2772.42 GOPS
M=2048, K=4096, N=8192, 1c: 554.51 GOPS
M=2048, K=4096, N=8192, 8c: 2753.46 GOPS
```

这一步不是照搬 ACL kernel，而是学习 ACL 的 selector 思路：让调度先按 shape 分流。下一步更适合继续补“可估算 selector”，把当前硬阈值扩展为基于 `M/N/K/thread` 的简化 cost model。

## prepare / LHS pack 计时口径复测

按新的 benchmark 口径，本轮调整了两个 adapter：

| 库 | 新计时区包含 | 仍在计时区外 |
|---|---|---|
| ACL | `gemm.prepare(prep_pack)` + `gemm.run(run_pack)` | Tensor 创建、configure、workspace 分配、输入填充 |
| KleidiAI | LHS pack + fixed ukernel dispatch | RHS pack、输入填充、输出分配 |
| i8gemm | `i8gemm_mt_dispatch`，内部包含 A reorder/pack + kernel | B pack、输入 pad/填充 |

注意：ACL 为了把 prepare 纳入计时，每个 timed run 都重新创建 dispatch 对象并绑定自己的 workspace；workspace 分配仍不计时，计时区从 `prepare()` 开始。KleidiAI 每个 rep 都重新执行 LHS pack，然后使用预打包 RHS 运行 ukernel。

复测输出：

```text
results/m8/prepare_lhs_timed_spot.csv
```

I8 代表点，单位 GOPS：

| Shape | Threads | i8gemm SVE | i8gemm NEON | ACL prepare timed | KleidiAI LHS timed |
|---|---:|---:|---:|---:|---:|
| 64x512x512 | 1 | 467.23 | 503.66 | 520.42 | 367.75 |
| 64x512x512 | 8 | 2075.71 | 2630.86 | 2045.39 | 890.25 |
| 512x1024x1024 | 1 | 528.55 | 369.31 | 506.19 | 405.48 |
| 512x1024x1024 | 8 | 2684.83 | 2305.20 | 2434.00 | 1513.19 |
| 2048x4096x8192 | 1 | 540.22 | 287.46 | 495.77 | 297.94 |
| 2048x4096x8192 | 8 | 2688.38 | 1824.41 | 2706.66 | 1747.96 |

BF16 代表点，单位 GOPS：

| Shape | Threads | i8gemm SVE | i8gemm NEON | ACL prepare timed | KleidiAI LHS timed |
|---|---:|---:|---:|---:|---:|
| 64x512x512 | 1 | 258.84 | 171.03 | 235.28 | 237.86 |
| 64x512x512 | 8 | 1138.81 | 1044.15 | 1126.79 | 1002.36 |
| 512x1024x1024 | 1 | 281.76 | 164.97 | 249.86 | 231.19 |
| 512x1024x1024 | 8 | 1266.04 | 1048.47 | 1278.68 | 1063.16 |
| 2048x4096x8192 | 1 | 277.98 | 143.93 | 250.65 | 183.57 |
| 2048x4096x8192 | 8 | 1459.33 | 805.69 | 1355.04 | 1035.06 |

这个口径下的主要变化：

1. ACL prepare 计入后，中大 shape 仍然和 i8gemm SVE 同档；I8 大矩阵 8 核为 `2706.66` vs i8gemm SVE `2688.38`。
2. KleidiAI LHS pack 计入后下降明显，尤其 I8，因为 LHS 是从 fp32 动态量化/pack 到 qai8dxp；它更接近“动态输入 A、预打包 RHS”的推理口径。
3. i8gemm 口径本来就把 A reorder/pack 放在 dispatch 内，因此这次相对更公平；B/RHS pack 三方仍然都没有纳入计时。

## 不计 pack 的 I8 kernel 对比

为了回答“不考虑 pack 时三方是什么水平”，本轮又补了一组 no-pack spot benchmark：

```text
results/m8/i8_no_pack_prepacked_spot.csv
```

口径如下：

| 库 | 计时区 | 说明 |
|---|---|---|
| i8gemm | `i8gemm_k_reo_ld` direct `.S` path | A/B 都已在计时外 pack；直接走 prepacked-A NEON 8x8 kernel |
| ACL | `gemm.run(run_pack)` | `prepare()` 在计时外；ACL selector 自动选 SVE/NEON kernel |
| KleidiAI | fixed ukernel dispatch | LHS/RHS pack 都在计时外；I8 ukernel 为 `qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm` |

I8 代表点，单位 GOPS：

| Shape | Threads | i8gemm NEON prepacked-A | ACL prepare-outside | KleidiAI pack-outside |
|---|---:|---:|---:|---:|
| 64x512x512 | 1 | 435.82 | 515.17 | 417.54 |
| 64x512x512 | 2 | 721.84 | 850.07 | 708.47 |
| 64x512x512 | 4 | 1266.88 | 1405.31 | 1188.87 |
| 64x512x512 | 8 | 2142.23 | 2045.19 | 1178.83 |
| 512x1024x1024 | 1 | 350.91 | 514.46 | 441.91 |
| 512x1024x1024 | 2 | 618.94 | 846.02 | 721.63 |
| 512x1024x1024 | 4 | 1145.12 | 1509.18 | 1277.78 |
| 512x1024x1024 | 8 | 2050.30 | 2610.14 | 2274.05 |
| 2048x4096x8192 | 1 | 297.13 | 503.48 | 284.29 |
| 2048x4096x8192 | 2 | 618.72 | 842.19 | 531.42 |
| 2048x4096x8192 | 4 | 1050.12 | 1507.47 | 984.39 |
| 2048x4096x8192 | 8 | 1787.05 | 2756.16 | 1823.86 |

结论：

1. 小/中等 shape 上，i8gemm 的 8x8 NEON prepacked-A kernel 本身并不弱。`64x512x512, 8c` 甚至略高于 ACL 的自动选择结果，说明原 8x8 kernel 的计算资源利用率确实比较充分。
2. 中大和超大 shape 上，差距主要不再来自 pack 成本，而来自 kernel 形态和调度。ACL 自动选择 `sve_interleaved_s8s32_mmla_8x3VL` 或 `sve_hybrid_s8s32_mmla_6x4VL`，在大 N/K 下吃到 SVE 宽向量、panel 组织和多线程切分收益。
3. KleidiAI 在 no-pack 口径下恢复明显，`512x1024x1024, 8c` 达到 `2274.05 GOPS`，但大 shape 仍略低于 ACL；它固定跑 NEON I8MM 16x4 ukernel，没有 ACL 那种按 shape 切 SVE/hybrid/interleaved 的 selector。
4. i8gemm 的 no-pack 大 shape 与 KleidiAI 接近，但 1 核明显低于 ACL。这说明继续优化时，单纯减少 A pack 开销不够，重点应该转到 SVE I8 kernel 的计算块、B panel 数据流、以及更细粒度的 N/M split selector。

补充：`i8gemm_k_ld` 不是完全 no-pack；它没有外部 A pack，但会在第一个 N slice 内部读原始行主序 A 并写 `A_reorder`，后续 N slice 再复用。因此真正“pack 不计时”的对比应该看 `i8gemm_k_reo_ld` 这组。

## 参考 ACL 的多线程调度优化

本轮对 ACL 的调度方式做了拆解。ACL runtime 里有一个通用的 `split_2d(max_threads, m, n)` 思路：当 kernel window 同时有 X/Y 两个维度时，线程会按问题规模比例拆到两个维度，而不是固定只按 M 或 N 切。这个思路不能直接照搬到 i8gemm 的 NEON 8x8 kernel 上，因为当前 kernel 对长 N sweep 的顺序复用很强；如果机械 2D split，会缩短每个任务的连续 N 区间，反而降低吞吐。

本轮改动：

1. 在 NEON I8 dispatch 中新增 `I8_GEMM_SPLIT=2d` 实验路径：强制 A prepack 一次，然后按 M block 和 N panel 做 2D 并行切分。
2. 默认 auto 不启用 2D split，因为 spot probe 显示它不稳定。
3. 把 ACL selector 的核心思想落到更保守的 split 修正：对 `M=128..256, K>=512, N<=8*M, threads>=4` 的 NEON I8 shape，避免旧规则误选 N-split，改为 M-split。

验证：

```text
All passed.
```

探针输出：

```text
results/m8/i8_neon_aclstyle_2d_probe.csv
results/m8/i8_neon_split_probe_after_2d.csv
results/m8/acl_sched_three_lib_compare.csv
```

关键 split 探针，单位 GOPS：

| Shape | Threads | M-split | N-split | 2D split | 结论 |
|---|---:|---:|---:|---:|---|
| 128x512x1024 | 4 | 1358.4 | 926.2 | 909.2 | M-split 最好 |
| 128x512x1024 | 8 | 2110.9 | 639.7 | 646.3 | M-split 最好 |
| 256x512x1024 | 4 | 1348.3 | 1314.1 | 1265.9 | M-split 略好 |
| 256x512x1024 | 8 | 2228.8 | 551.6 | 1157.0 | M-split 最好 |
| 512x1024x1024 | 8 | 2126.7 | 2262.2 | 2299.1 | N/2D 略好 |
| 2048x4096x8192 | 8 | 1882.4 | 1880.8 | 1850.5 | M/N 接近，2D 略差 |

新 auto 规则命中的收益：

| Shape | Threads | 旧 auto 等价 N-split | 新 auto | 提升 |
|---|---:|---:|---:|---:|
| 128x512x1024 | 4 | 926.2 | 1302.1 | 1.41x |
| 128x512x1024 | 8 | 639.7 | 2256.3 | 3.53x |
| 256x512x1024 | 4 | 1314.1 | 1328.8 | 1.01x |
| 256x512x1024 | 8 | 551.6 | 2305.4 | 4.18x |

三方复测使用 prepare/LHS pack 计时口径，I8 代表点如下，单位 GOPS：

| Shape | Threads | i8gemm SVE | i8gemm NEON | ACL | KleidiAI |
|---|---:|---:|---:|---:|---:|
| 64x512x512 | 8 | 1809.7 | 2218.8 | 2299.7 | 724.2 |
| 128x512x1024 | 8 | 2385.6 | 2256.3 | 2509.4 | 1445.8 |
| 256x512x1024 | 8 | 2606.5 | 2305.4 | 2428.1 | 1474.1 |
| 512x1024x1024 | 8 | 2554.9 | 2336.7 | 2601.2 | 1529.0 |
| 2048x4096x8192 | 8 | 2684.8 | 1841.2 | 2703.4 | 1750.5 |

结论：

1. ACL 的“按 shape 选择调度方式”值得学习，但 i8gemm NEON 8x8 不能直接套 2D split。2D 在部分 `512x1024x1024` 点有轻微收益，但在 `64x512x512`、`128x512x1024` 和超大 shape 上会回退。
2. 真正有效的是修正 N-split 触发条件：中等 M、宽 N、4/8 核时，旧规则被 packed-B 大小误导，选了重复扫 M 的 N-split；M-split 能保持每线程长 N sweep，吞吐更稳定。
3. 新规则后，`128x512x1024` 和 `256x512x1024` 的 NEON 8 核性能从旧 N-split 的低谷恢复到 `2.2~2.3 TOPS`。这让 i8gemm NEON 在这些中等 shape 上重新接近 ACL，甚至 `256x512x1024, 8c` 高于本轮 ACL 结果。
4. 大 shape 上 i8gemm SVE 仍是主力，`2048x4096x8192, 8c` 为 `2684.8 GOPS`，和 ACL `2703.4 GOPS` 基本同档；NEON 由于计算块和向量宽度限制仍明显落后。

## SVE I8 N3 interleaved kernel 实验

本轮参考 ACL 的 `sve_interleaved_s8s32_mmla_8x3VL`，在 i8gemm SVE 下增加了一个实验性的 `8x3VL` int32 kernel，入口为 `i8gemm_k_nld_n3`，通过 `I8_SVE_N3=1` 打开。默认路径不启用。

改动内容：

1. B pack 从旧的 `2VL` panel 扩展为 N3 布局：每个 128-bit segment 存 `12` 个输出列，对应 `6` 个 column-pair vector。
2. kernel 计算块从旧 SVE 主路径的 `8x2VL` 扩到 `8x3VL`，使用 `z8..z31` 共 24 个 int32 累加寄存器。
3. 写回 offset 增加 N3 专用版本：每个 128-bit segment 的列跨度从旧路径的 `8` 改成 `12`，尾行继续用 predicate 屏蔽。
4. `I8_SVE_N3=1` 时强制启用 A reorder，并绕开现有 M12 路径，避免旧 `nld1/nld2/nld4/M12` kernel 读取 N3 B pack。
5. benchmark 的 SVE padding 也识别 `I8_SVE_N3=1`，否则 dispatch 级测试会按旧 `2VL` tile padding。

验证：

```text
./i8gemm/tests/build/test_correctness_sve
SVE correctness OK: 6000 wrapper+dispatch cases

I8_SVE_N3=1 /tmp/test_i8_sve_n3_i32
i8 SVE N3 int32 wrapper correctness OK
```

注意：N3 目前只迁移了 int32 zero-C 路径。i8 的 fp32/bias 输出 kernel 仍然按旧 B pack 读取，所以不能在这些路径上打开 `I8_SVE_N3=1`。

性能探针见：

```text
results/acl_sched_sve_n3_experiment.csv
```

单位 GOPS：

| Shape | Threads | 默认 SVE | N3 实验 | N3/默认 |
|---|---:|---:|---:|---:|
| 64x512x512 | 1 | 453.7 | 446.2 | 0.98x |
| 64x512x512 | 2 | 715.8 | 763.4 | 1.07x |
| 64x512x512 | 4 | 1214.2 | 1168.8 | 0.96x |
| 64x512x512 | 8 | 44.0 | 1878.5 | 异常点，需复测 |
| 128x512x1024 | 1 | 480.3 | 456.1 | 0.95x |
| 128x512x1024 | 8 | 2419.4 | 2335.8 | 0.97x |
| 256x512x1024 | 8 | 2553.6 | 2283.0 | 0.89x |
| 512x1024x1024 | 8 | 2776.2 | 2633.2 | 0.95x |
| 2048x4096x8192 | 8 | 2620.3 | 2519.2 | 0.96x |

结论：

1. 只把 N panel 从 `2VL` 扩到 `3VL`，并不会自动得到 ACL 的收益。本轮 N3 kernel 多数点慢 `3%~11%`。
2. 原因是 ACL 的 `8x3VL` 不是单纯“更宽”：它的 K loop 有更深的 load/compute 交错、跨 slice 预取和更细的汇编调度；本轮 N3 版本仍是较直接的 `load 6B + load A + 24 smmla` 排布，load-use 距离和 B/A load 带宽压力都更差。
3. 现有 i8gemm M12 路径是 `12x2VL`，寄存器已经吃满 24 个 accumulator；ACL 的 `6x4VL` 是另一种取舍，用更少 M 换更宽 N。若继续追 ACL，应新增独立 `6x4VL` hybrid/interleaved kernel，而不是在 `12x2VL` 上硬扩 N。
4. 当前 N3 更适合作为后续 ACL-style 调度实验基线，不应设为默认路径。下一步更有价值的是做 `6x4VL` kernel、K32 深展开、以及把 N tail 从 padding-only 改成 ACL 那种真实 predicate tail。
