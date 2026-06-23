# i8gemm / ACL / KleidiAI M8 Parts Kernel Dispatch Analysis

## 测试范围

本次覆盖 `run_m8_parts.sh` 的完整规模网格：

- `K = 128 256 512 1024 2048 4096`
- `M = 16 32 64 128 256 512 1024 2048`
- `N = 16 32 64 128 256 512 1024 2048 4096 8192`
- 共 480 个 shape，每个 shape 测 `1/2/4/8` 核、`bf16/i8`、`i8gemm_sve/i8gemm_neon/ACL/KleidiAI`

输入格式统一为 A row-major、B 预打包、C row-major，C 初值为 0。CPU 绑定使用 `GOMP_CPU_AFFINITY=0-7`、`OMP_PLACES=cores`、`OMP_PROC_BIND=close`。完整覆盖结果为 `../m8/three_lib_grid_full_reps1.csv`，共 15360 条，全部 `ok`。另跑代表点 warmup/repeat 结果为 `../m8/three_lib_grid_quick_steady.csv`。

限制条件：ACL 的 I8 路径是 `int8 x int8 -> int32`，i8gemm 的 I8 路径也是 int32 输出；KleidiAI 当前可用于本机 NEON I8MM 的固定微核是 `qai8dxp/qsi8cxp -> f32` dequant/clamp 输出，内部使用 `smmla`，但不是裸 int32 输出。BF16 三方路径均是 BF16 输入、FP32 accumulate/output。

## 打桩确认

ACL adapter 在每条 CSV 的 `note` 写入 `get_compatible_kernels()` 的 default kernel。实测选择只落在 MMLA kernel：

- BF16: `a64_hybrid_bf16fp32_mmla_6x16`、`a64_interleaved_bf16fp32_mmla_8x12`、`sve_hybrid_bf16fp32_mmla_6x4VL`、`sve_interleaved_bf16fp32_mmla_8x3VL`
- I8: `a64_hybrid_s8s32_mmla_6x16`、`a64_interleaved_s8s32_mmla_8x12`、`sve_hybrid_s8s32_mmla_6x4VL`、`sve_interleaved_s8s32_mmla_8x3VL`

ACL 源码证据：BF16 MMLA dispatch 在 `ComputeLibrary/src/core/NEON/kernels/arm_gemm/gemm_bf16.cpp:136` 和 `:184`；I8 MMLA dispatch 在 `ComputeLibrary/src/core/NEON/kernels/arm_gemm/gemm_int8.cpp:107` 和 `:132`。BF16 SVE microkernel 中有 `bfmmla`，I8 SVE microkernel 中有 `smmla`。

KleidiAI adapter 直接调用固定微核，并把固定微核名写入 `note`：

- BF16: `kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla`，源码含 `bfmmla` 指令。
- I8: `kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm`，源码含 `smmla` 指令。

本次新增/修改的测试入口：

- `i8gemm/tests/run_three_lib_grid.py`
- `i8gemm/tests/bench_acl_dispatch.cpp`
- `i8gemm/tests/bench_kleidiai_dispatch.c`

## 代表点结果

单位是 GOPS，取 `three_lib_grid_quick_steady.csv`。

### BF16

| Shape | Threads | i8gemm SVE | i8gemm NEON | ACL | KleidiAI |
|---|---:|---:|---:|---:|---:|
| M16 K128 N128 | 1 | 133.0 | 156.2 | 222.3 | 195.8 |
| M16 K128 N128 | 8 | 133.7 | 280.3 | 236.8 | 216.0 |
| M64 K512 N512 | 1 | 249.9 | 169.9 | 245.5 | 238.2 |
| M64 K512 N512 | 8 | 1017.0 | 952.9 | 1237.3 | 1226.3 |
| M512 K1024 N1024 | 1 | 282.0 | 165.0 | 253.0 | 236.7 |
| M512 K1024 N1024 | 8 | 1433.4 | 1026.4 | 1280.0 | 583.5 |
| M2048 K4096 N8192 | 1 | 277.6 | 143.8 | 248.1 | 180.9 |
| M2048 K4096 N8192 | 8 | 1439.8 | 791.3 | 1365.0 | 1040.4 |

### I8

| Shape | Threads | i8gemm SVE | i8gemm NEON | ACL | KleidiAI |
|---|---:|---:|---:|---:|---:|
| M16 K128 N128 | 1 | 138.9 | 252.7 | 380.3 | 258.7 |
| M16 K128 N128 | 8 | 86.8 | 213.6 | 233.5 | 197.9 |
| M64 K512 N512 | 1 | 457.9 | 403.1 | 507.9 | 419.5 |
| M64 K512 N512 | 8 | 1011.1 | 2151.8 | 2087.2 | 1166.8 |
| M512 K1024 N1024 | 1 | 537.1 | 341.8 | 512.6 | 439.4 |
| M512 K1024 N1024 | 8 | 2671.2 | 2238.4 | 2696.7 | 2186.0 |
| M2048 K4096 N8192 | 1 | 550.8 | 285.8 | 492.9 | 300.3 |
| M2048 K4096 N8192 | 8 | 2718.1 | 1884.5 | 2729.8 | 1775.8 |

## 全网格概览

完整 480 shape 单次覆盖的中位数显示 ACL 在多数小/中 shape 上胜出，尤其多核时更明显：

| DType | Threads | i8gemm SVE | i8gemm NEON | ACL | KleidiAI |
|---|---:|---:|---:|---:|---:|
| BF16 | 1 | 204.8 | 133.5 | 220.2 | 203.0 |
| BF16 | 2 | 272.3 | 175.3 | 367.3 | 267.8 |
| BF16 | 4 | 302.2 | 201.2 | 622.0 | 299.0 |
| BF16 | 8 | 241.4 | 181.0 | 836.3 | 237.0 |
| I8 | 1 | 318.8 | 266.5 | 429.0 | 339.4 |
| I8 | 2 | 343.2 | 302.8 | 695.8 | 395.3 |
| I8 | 4 | 348.0 | 317.0 | 1109.5 | 363.1 |
| I8 | 8 | 257.3 | 234.5 | 1310.1 | 250.5 |

按 shape 胜出次数看，ACL 在多核档占绝对多数：BF16 8 核 428/480，I8 8 核 426/480。i8gemm SVE 的优势主要集中在较大 M/K/N、计算量足够大时；代表点中 BF16 大 shape 和 I8 大 shape 可以达到或略超过 ACL。

## 差异原因

ACL 的优势来自两个点。第一，它有完整的 kernel selector，会根据 shape 在 hybrid/interleaved、A64/SVE MMLA kernel 间切换；小 M 或窄 N 更容易走 hybrid，宽 N/大 K 走 interleaved。第二，ACL 的 `NEScheduler` 和 assembly GEMM 的 workspace/pack 管理比较成熟，多核下能把 pack、kernel work 和 cache locality 平衡得更稳，所以完整网格中 2/4/8 核中位数明显领先。

i8gemm 的 SVE 大 shape 很强，原因是自有 SVE BF16/I8 kernel 针对大矩阵有更直接的 tiled compute path，且 I8/BF16 调度里有 M split、N split、2D split、nblock split 等策略。大矩阵时调度开销被计算量摊薄，SVE `bfmmla/smmla` 吞吐能体现出来；小 M、小 N 时线程 clamp、pack A、尾块和调度分支占比上升，所以全网格中位数被大量小 shape 拉低。

i8gemm NEON 在 I8 的 `M64 K512 N512` 8 核点超过 ACL，是因为这个 shape 对 NEON 8x tile 和 M/N split 比较友好；但 BF16 和大 shape 上 NEON 受 128-bit 向量宽度限制，通常落后于 SVE MMLA。

KleidiAI 的单核 BF16/I8 微核本身质量不错，单核中位数接近 i8gemm SVE/ACL；但本次 adapter 是直接固定微核加 OpenMP 外层按 M block 静态切分，没有 ACL 那样的 shape selector、N/K block 调度和 workspace scheduler。M 很小时可并行 work unit 不足，大 N/大 K 时只按 M 切分也会让 packed RHS 复用和负载均衡不如 ACL。KleidiAI I8 还多了 dequant/clamp 到 f32 输出的路径，和 int32-output 对比不是完全同口径。

## 结论

如果目标是“默认库接口、多 shape、多核稳态”，ACL 当前最稳，且打桩确认 BF16/I8 都走 MMLA 指令族。如果目标是“大矩阵、固定 SVE kernel 极限吞吐”，i8gemm SVE 已经能在若干大 shape 上追平或超过 ACL，尤其 I8 8 核在 `M2048 K4096 N8192` 达到 2718.1 GOPS，与 ACL 2729.8 GOPS 基本同档。KleidiAI 更适合当作固定微核基线；要公平比较完整库能力，需要再补一层和 ACL 类似的 shape selector 与 2D/N split 调度，并找到或实现 int32-output I8MM NEON/SVE 路径。

## Kernel 实现逻辑对比

### 总体结构

三套实现的最大区别不是指令是否用了 MMLA，而是“微核之外”的组织方式不同：

| 库 | Kernel 选择 | 主微核形态 | A 处理 | B 处理 | C 输出 | 多核边界 |
|---|---|---|---|---|---|---|
| i8gemm | 手写 SVE/NEON 两套入口，测试中显式选择 | 自有 assembly kernel，NEON 常见 8x8，SVE 使用 VL-aware tile | A 行主序输入，kernel/外层按 M block 做 A reorder/pack | B 由库外/上层 pack 成 kernel 友好布局 | I8 为 int32 row-major，BF16 为 f32 row-major | 自有 OpenMP 调度，按 M/N/2D/nblock 等策略拆 |
| ACL | `arm_gemm` selector 自动选择 hybrid/interleaved、SVE/A64 | assembly framework kernel，SVE interleaved 8x3VL，SVE hybrid 6x4VL 等 | 由 GEMM framework 管理 pack/indirect/direct input | framework 内部 pack 或 workspace 管理 | I8 int32，BF16 f32 | `NEScheduler` + GEMM strategy 自带调度 |
| KleidiAI | 本次固定调用一个 ukernel | NEON fixed ukernel，BF16 8x12，I8 16x4 | 调用前必须按 ukernel API pack LHS | 调用前必须按 ukernel API pack RHS | BF16 f32；I8 本次为 f32 dequant/clamp | adapter 外层 OpenMP 只按 M block 静态切 |

因此，单核看的是“微核 + pack 格式”的效率；多核看的是“微核 + shape selector + work partition + workspace/pack 复用”的综合效率。ACL 多核中位数强，主要强在后半部分；i8gemm 大 shape 强，主要强在 SVE kernel 和大矩阵调度路径；KleidiAI 单核常常接近，但本次缺少完整库级调度。

### i8gemm kernel 逻辑

i8gemm 的 NEON I8/BF16 kernel 是比较直接的 M-N-K 循环。源码注释明确说明循环顺序是最外层 M、中间 N、最内层 K；每个 M block 的第一个 N iteration 从原始 A 跨 stride 读取并写入 A reorder buffer，后续 N iteration 直接复用 reordered A。这样做的好处是 B 已经 pack 好，N 方向扫多个 tile 时 A 的访存变成连续读取；代价是小 shape 下 A reorder 和尾块处理占比会比较明显。

NEON BF16 kernel 以 8x8 输出块组织，`bfmmla` 的 accumulator 布局是 2 行 x 2 列为一个 vector lane group；I8 NEON 也是 8x8 输出块，`smmla` accumulator 采用类似的 2x2 blocked 布局。为了把 row-major C 和 MMLA accumulator 布局互转，NEON kernel 里有大量 `zip1/zip2` 的 C load/store 重排。这种布局对固定 8x8 主块很有效，但遇到 M tail 需要额外的 1/2/4 行专门路径。

SVE 版本的 i8gemm 更偏向 VL-aware kernel。I8 SVE 用 `smmla z*.s, z*.b, z*.b`，BF16 SVE 用 `bfmmla z*.s, z*.h, z*.h`。SVE I8 kernel 里有 `ld1rqb` 复制 A panel、`ld1b` 读取 B panel，然后对 z16-z31 做 accumulator；store 侧用 predicate 处理尾列，并提供 int32 store、f32 convert store、bias f32 store 等路径。BF16 SVE kernel 同样用 z16-z31 accumulator，并根据是否 load C、是否 bias 初始化 accumulator。

外层调度方面，i8gemm 不是只有一个简单 parallel-for。I8 有线程数 clamp、小 M 时倾向 N split、大 M 时可 M split；BF16 SVE 有 nblock split、ngroup split、2D split、M12/M8 不同 block 策略。这个设计解释了两个现象：大 shape 时 i8gemm SVE 可以很好地吃满计算；小 shape 或线程过多时，调度判断、pack/reorder、tail kernel 的成本会让中位数下降。

### ACL kernel 逻辑

ACL 的关键是 `arm_gemm` strategy table。BF16 会在 `sve_interleaved_bf16fp32_mmla_8x3VL`、`sve_hybrid_bf16fp32_mmla_6x4VL`、`a64_interleaved_bf16fp32_mmla_8x12`、`a64_hybrid_bf16fp32_mmla_6x16` 等 MMLA kernel 间选择；I8 会在 `sve_interleaved_s8s32_mmla_8x3VL`、`sve_hybrid_s8s32_mmla_6x4VL`、`a64_interleaved_s8s32_mmla_8x12`、`a64_hybrid_s8s32_mmla_6x16` 等路径间选择。选择依据包含 CPU feature、K size、shape 估算周期等。

ACL 的 interleaved kernel 更适合大 K、大 N、高复用场景，典型 SVE interleaved 形态是 8x3VL；hybrid kernel 更适合小 M 或不规则输入，典型 SVE hybrid 形态是 6x4VL。hybrid kernel 的逻辑更复杂，会处理 input pointer array、string loop、indirect/direct input、bias/accumulate 标志等，因此小 shape 下比固定大块 kernel 更稳。

从 assembly 片段看，ACL SVE MMLA kernel 的主循环把 load 和 MMLA 指令交错排布，例如 I8 interleaved kernel 连续执行 `smmla` 并穿插 `ld1rqb/ld1b`，BF16 interleaved kernel 同理使用 `bfmmla` 并穿插 `ld1rqh/ld1h`。它的目标是隐藏 load latency，并让 z8-z31 等 accumulator/register 资源维持高利用。相比 i8gemm，ACL 单个 kernel 的通用框架更重，但 selector 能把 shape 分发到更合适的 family。

调度上，ACL 使用 `CpuGemmAssemblyDispatch` 和 `NEScheduler`。pack、workspace、kernel run 被框架统一管理；这也是为什么全网格多核下 ACL 胜率高。它不只是微核快，而是对小/中/大 shape 都有相对合适的 kernel family 和工作切分。

### KleidiAI kernel 逻辑

KleidiAI 的接口更像“固定 ukernel building block”。BF16 微核 `kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla` 的计算块是 8x12：LHS pack 成 bf16p8x4，RHS pack 成 bf16p12x4 加 bias f32，主循环直接在 NEON register 上执行大量 `bfmmla`，最后按 row-major f32 store 并支持 clamp。

I8 微核 `kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm` 的计算块是 16x4：LHS 是动态量化/packed 的 qai8dxp，RHS 是 qsi8cxp，RHS pack 中包含 scale、row/reduction sum 和 bias 等元数据。主循环用 NEON `smmla` 做 int32 accumulate，但输出阶段会应用 scale/dequant/clamp，最终写 f32。因此它证明了 I8MM/SMMLA 微核能力，但本次不是 int32-output 同口径。

KleidiAI 的优势是微核 API 清晰、固定格式下开销低，单核小缓存区间表现很好；缺口是它本身不替调用者做“完整 GEMM 库”级别的 shape selector 和多维调度。本次 adapter 只在 M block 上做 OpenMP static split，B panel 全局复用但没有 N split/K split/2D split，因此多核和宽 N 场景很难达到 ACL 这种成熟调度的稳定性。

### 逻辑差异对应到性能现象

单核时，KleidiAI 和 ACL 在小 L1/H2 区间经常领先，因为固定 ukernel 或 hybrid kernel 的入口成本和尾块处理更适合小 shape；i8gemm SVE 在大 GT_L2 区间更容易超过 ACL BF16，因为大矩阵能摊薄 pack/reorder 和调度开销，SVE kernel 的直接吞吐占主导。

多核时，ACL 胜率变高，是因为它把 kernel family、workspace、pack、scheduler 作为整体优化；i8gemm 虽然也有 M/N/2D split，但更偏当前项目的特定优化路径，大量小 shape 会被调度和尾块成本拉低；KleidiAI 本次固定 M split 的 adapter 缺少 N/K 方向并行，M 小时线程数很难有效利用，M 大但 N/K 很宽时也不如 ACL 的策略丰富。

一句话概括：i8gemm 是“项目内定制高吞吐 kernel + 自有调度”，ACL 是“多 kernel family 自动选择 + 成熟通用调度”，KleidiAI 是“高质量固定 ukernel primitives，需要调用者自己补完整调度层”。

### I8 NEON: 8x8 拉满但仍可能慢于 KleidiAI 的原因

只看 I8 NEON/I8MM，三方主计算块并不一样：

| 库 | NEON I8 kernel | 主计算块 | 输出 |
|---|---|---:|---|
| i8gemm | `i8gemm_k.S` | 8x8 | int32 row-major |
| ACL interleaved | `a64_interleaved_s8s32_mmla_8x12` | 8x12 | int32 row-major |
| ACL hybrid | `a64_hybrid_s8s32_mmla_6x16` | 6x16 | int32 row-major |
| KleidiAI | `kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm` | 16x4 | f32 row-major, dequant/clamp 后写回 |

i8gemm 的 8x8 微核内部 `smmla` 可以排得很满，但“计算资源拉满”只覆盖主 compute loop，并不覆盖完整 GEMM 的数据流成本。完整路径还包括 A reorder、C row-major 和 `smmla` blocked accumulator 之间的 `zip1/zip2` 重排、tail kernel、线程切分、cache/TLB 行为等。

KleidiAI 的 16x4 是高瘦块。对 `M` 很大但 `N` 较窄的 shape，例如 `M1024/2048, N16/32/64`，它沿 M 方向更容易形成连续工作量，输出列块只有 4 列，store pattern 更贴近窄 N 场景；i8gemm 的 8x8 则在 N 方向 tile 较少时，A reorder 复用次数变少，C 重排和调度开销更难被摊薄。因此 GT_L2 完整集合里，KleidiAI 在不少“大 M、窄 N”点上会领先。

但这不是说 KleidiAI 在所有大规模都更快。代表点 `M2048 K4096 N8192` 中，I8 NEON repeat 结果为：1 核 i8gemm 285.8 GOPS、KleidiAI 300.3 GOPS；2 核 i8gemm 578.8、KleidiAI 536.6；4 核 i8gemm 1044.0、KleidiAI 987.1；8 核 i8gemm 1884.5、KleidiAI 1775.8。也就是说，当 N 足够宽、A reorder 能被大量 N tile 摊薄时，i8gemm 的 8x8 路径会重新占优。

所以更准确的结论是：i8gemm 8x8 的 compute loop 利用率高，适合规则且 N 足够宽的矩阵；KleidiAI 16x4 更适合大 M、窄/中 N 的固定 ukernel 场景。两者差异主要来自计算块形状和数据流，而不只是 `smmla` 指令本身。

### KleidiAI I8 结果的存储格式

KleidiAI 的结果确实是按 row-major 写回的，不是 packed C。它的 dst offset 逻辑是：

```text
offset = n_idx * sizeof(float) + m_idx * dst_stride_row
```

本次 adapter 调用时传入：

```text
dst = base + m_idx * N
dst_stride_row = N * sizeof(float)
dst_stride_col = sizeof(float)
```

因此输出地址就是 `C[m][n] = base + m * N + n`，也就是标准 row-major。微核内部完整 4 列块用 `str q*` 连续写一行的 4 个 float，然后 `add x20, x20, dst_stride_row` 切到下一行；尾列用 `st1 {v*.d}` / `st1 {v*.s}` 写 1/2/3 个 float，也保持 row-major。

需要强调的是存储布局和数值类型是两件事：KleidiAI I8 这里的布局是 row-major，但数值不是 int32 accumulator 原样写回，而是在 `smmla` int32 accumulate 后应用 scale、dequant 和 clamp，最终以 float32 写回。也就是说它是 **f32 row-major 输出**，不是 **int32 row-major 输出**。

## i8gemm 参考 KleidiAI 的优化复测

KleidiAI I8 这套路径的核心不是单独的 `16x4` 形状，而是一整套匹配的数据流。它的 ukernel 是 `M step=16, N step=4, kr=8`，LHS 按 4 行一组打包，4 个 `mr=4` 子块拼成 16 行；RHS 按 4 列一组打包，K 内部按 32 对齐。主循环里 LHS/RHS 都是连续 packed load，手写 assembly 把 `ldr` 和 `smmla` 交错排布，最后做 dequant/clamp 并按 row-major 写 f32。也就是说，KleidiAI 的优势来自 `packed LHS + packed RHS + 手写调度 + 固定 store 语义 + 简单外层切分` 的组合。

本轮在 i8gemm 里落地了两类低风险优化。第一，8x8 NEON int32 路径增加 `zero-C` kernel 入口：`i8gemm_k_zero*` 和 `i8gemm_k_reo_zero*`，默认跳过 C load，从 0 accumulator 开始；如果需要旧的累加语义，可以设置 `I8_GEMM_ACCUMULATE_C=1` 回退。第二，16x4 实验路径 `i8gemm_mt_dispatch_m16n4` 增加 M16/K8 packed-A：每个 M16 block 先把 A 打包一次，然后所有 N4 panel 复用，不再为每个 N4 panel 重读 row-major A。为了保持 i8gemm API，B 仍复用现有 `i8_pack_B` 的 8 列格式，C 仍输出 int32 row-major。

正确性验证：`tests/test_i8_m16n4_correctness.c` 通过，覆盖整块、M/N/K tail 和 1/2/4 线程。完整 480 shape x 4 线程复测输出为 `results/m8/three_lib_grid_m16n4_opt_full_reps1.csv`，共 17280 行，全部 `status=ok`。

### 全网格中位数

| Threads | i8gemm NEON 8x8 | i8gemm NEON m16n4 | KleidiAI I8 NEON I8MM |
|---:|---:|---:|---:|
| 1 | 282.1 | 217.0 | 340.3 |
| 2 | 315.0 | 279.3 | 383.9 |
| 4 | 319.3 | 310.2 | 367.9 |
| 8 | 246.0 | 237.6 | 255.8 |

全网格包含大量很小的 shape，线程启动、pack 开销和 tail 处理会放大。优化后 m16n4 比上一版明显更稳，但总体中位数仍低于 KleidiAI；8x8 由于 zero-C 路径，单核和双核中位数也有提升。

### GT_L2 大规模中位数

| Threads | i8gemm NEON 8x8 | i8gemm NEON m16n4 | KleidiAI I8 NEON I8MM |
|---:|---:|---:|---:|
| 1 | 343.8 | 245.7 | 364.4 |
| 2 | 545.4 | 402.7 | 544.8 |
| 4 | 832.0 | 666.2 | 862.5 |
| 8 | 985.2 | 885.2 | 1040.8 |

和上一版 m16n4 相比，GT_L2 中位数从 `170.0/311.5/537.1/789.7` 提升到 `245.7/402.7/666.2/885.2`。主要收益来自 packed-A 复用：N 越宽，同一块 A 被更多 N4 panel 复用，越能摊薄 A pack 成本。相对默认 8x8，m16n4 的 GT_L2 中位比例现在为 `74.0%/75.9%/81.1%/88.2%`；相对 KleidiAI 为 `70.5%/74.1%/76.8%/81.4%`。

### 代表点

| Shape | Threads | i8gemm NEON 8x8 | i8gemm NEON m16n4 | KleidiAI I8 NEON I8MM |
|---|---:|---:|---:|---:|
| 64x512x512 | 1 | 286.3 | 234.8 | 413.0 |
| 64x512x512 | 8 | 122.7 | 124.8 | 111.1 |
| 512x1024x1024 | 1 | 347.8 | 304.1 | 440.0 |
| 512x1024x1024 | 8 | 1443.9 | 1121.9 | 1266.0 |
| 2048x4096x8192 | 1 | 339.9 | 227.0 | 287.9 |
| 2048x4096x8192 | 8 | 1909.8 | 1496.8 | 1793.3 |

代表点能看到两种效果：m16n4 对上一版提升很大，例如 `2048x4096x8192, 8c` 从约 `1067.4` 提到 `1496.8 GOPS`；但默认 8x8 在超宽 N 大矩阵上仍更强，因为它的 8 列 B panel 和 8x8 asm 已经高度匹配，A reorder 也能被大量 N tile 摊薄。

### 仍然存在的差距

第一，m16n4 仍是 C/ACLE microkernel。虽然 `vmmlaq_s32` 会生成 `smmla`，但 load 提前、K loop 展开、寄存器分配、store 排布都由编译器决定，达不到 KleidiAI 手写 assembly 那种稳定的 `ldr/smmla/ldr/smmla` 交错。

第二，m16n4 仍使用 i8gemm 8 列 `B_reo`。对 N4 kernel 来说，这是从 8 列 panel 中取半块使用，RHS 的 panel stride、预取粒度和 cache footprint 都不是为 N4 专门设计的。要继续逼近 KleidiAI，需要补 N4 专用 B pack。

第三，当前 16x4 的 packed-A 只保存 int8 值，不包含 KleidiAI 为量化输出准备的 multiplier/zero-point 元数据；这符合 i8gemm 的 int32 输出目标，但也意味着不能直接复用 KleidiAI 的完整 ukernel store/dequant 路径。

第四，调度仍是简单 OpenMP static `(M16 block, N4 block)`。这比只按 M 切分更容易暴露并行度，但还没有按 shape 选择 M/N/2D split、K split、workspace 复用和亲和策略。后续如果要把 16x4 纳入默认 selector，建议只在窄 N、小 M tail、多线程 work unit 不足的场景启用；宽 N 大矩阵继续优先 8x8。

## 单核详细对比

本节只统计完整覆盖结果 `three_lib_grid_full_reps1.csv` 中 `threads=1` 的 480 个 shape。单核结果更能反映 microkernel 本身、pack 格式、尾块处理和 kernel selector 的差异，基本不受多线程调度影响。

### 单核中位数

| DType | i8gemm SVE | i8gemm NEON | ACL | KleidiAI |
|---|---:|---:|---:|---:|
| BF16 | 204.8 | 133.5 | 220.2 | 203.0 |
| I8 | 318.8 | 266.5 | 429.0 | 339.4 |

单核总体上 ACL 的中位数最高，BF16 上领先 i8gemm SVE 约 7.5%，I8 上领先 i8gemm SVE 约 34.6%。不过从胜出次数看，BF16 单核并不是 ACL 一家独大：ACL、i8gemm SVE、KleidiAI 三者都在不同规模段有明显优势区间。

### 按缓存规模分层

| DType | Cache | i8gemm SVE | i8gemm NEON | ACL | KleidiAI |
|---|---|---:|---:|---:|---:|
| BF16 | L1 | 35.0 | 48.1 | 50.0 | 68.7 |
| BF16 | L2 | 191.2 | 118.3 | 219.9 | 214.1 |
| BF16 | H2 | 130.7 | 97.0 | 157.3 | 176.8 |
| BF16 | GT_L2 | 235.6 | 146.0 | 232.1 | 210.3 |
| I8 | L1 | 75.0 | 89.8 | 68.1 | 133.8 |
| I8 | L2 | 318.8 | 256.7 | 462.9 | 358.6 |
| I8 | H2 | 229.3 | 196.2 | 294.8 | 317.3 |
| I8 | GT_L2 | 416.6 | 308.0 | 466.2 | 366.3 |

BF16 单核的分层特征比较清楚：KleidiAI 在 L1/H2 段更好，ACL 在 L2 段更稳，i8gemm SVE 在 GT_L2 大规模段略高于 ACL。原因是 KleidiAI 固定微核路径很短，小规模时没有复杂 selector/scheduler 成本；ACL 的 hybrid/interleaved 选择在中等规模更稳；i8gemm SVE 在大规模时 `bfmmla` 主循环占比最高，pack 和尾块成本被摊薄。

I8 单核上 ACL 的 L2/GT_L2 中位数最高，KleidiAI 在 L1/H2 更强。KleidiAI L1 的优势来自固定 `neon_i8mm` 微核路径短、开销小，但它输出 f32 dequant，和 int32-output 并非完全同口径。i8gemm SVE 在 GT_L2 能接近 ACL，但 L1/L2/H2 中位数偏低，说明当前 SVE I8 路径在小中规模上受 pack、tail 和调用路径影响更明显。

### 单核胜出次数

| DType | Cache | i8gemm SVE | i8gemm NEON | ACL | KleidiAI |
|---|---|---:|---:|---:|---:|
| BF16 | L1 | 0 | 0 | 3 | 34 |
| BF16 | L2 | 8 | 0 | 32 | 19 |
| BF16 | H2 | 1 | 1 | 29 | 71 |
| BF16 | GT_L2 | 156 | 0 | 110 | 24 |
| BF16 | ALL | 164 | 1 | 172 | 143 |
| I8 | L1 | 0 | 3 | 0 | 54 |
| I8 | L2 | 4 | 0 | 54 | 3 |
| I8 | H2 | 1 | 1 | 48 | 74 |
| I8 | GT_L2 | 83 | 0 | 146 | 18 |
| I8 | ALL | 88 | 1 | 248 | 143 |

BF16 单核的三个主力各有地盘：KleidiAI 赢小规模和 H2，i8gemm SVE 赢大规模，ACL 在 L2 和 GT_L2 都有稳定覆盖，所以总胜出次数最高但优势不悬殊。I8 单核则更偏 ACL：ACL 在 248/480 个 shape 胜出，主要来自 L2 和 GT_L2；KleidiAI 的 143 次胜出集中在 L1/H2；i8gemm SVE 的 88 次胜出主要集中在 GT_L2。

### 相对 ACL 的单核性能

| DType | Impl | Median Ratio vs ACL | P25 | P75 |
|---|---|---:|---:|---:|
| BF16 | i8gemm SVE | 0.952 | 0.817 | 1.075 |
| BF16 | i8gemm NEON | 0.643 | 0.508 | 0.770 |
| BF16 | KleidiAI | 0.969 | 0.884 | 1.068 |
| I8 | i8gemm SVE | 0.904 | 0.724 | 1.030 |
| I8 | i8gemm NEON | 0.696 | 0.556 | 0.835 |
| I8 | KleidiAI | 0.894 | 0.774 | 1.109 |

BF16 下 i8gemm SVE 和 KleidiAI 的单核中位性能都接近 ACL，且 P75 超过 1，说明它们在不少 shape 上可以超过 ACL。i8gemm NEON 的 ratio 明显低，主要是 NEON BF16 MMLA tile 固定、向量宽度和 SVE 相比吃亏。

I8 下 i8gemm SVE 单核中位约为 ACL 的 90.4%，KleidiAI 约为 89.4%，但 KleidiAI 的 P75 达到 1.109，说明固定 NEON I8MM 微核在部分小/中 shape 上效率很高。i8gemm NEON 约为 ACL 的 69.6%，在大多数 shape 上不是最优。

### 单核原因分析

ACL 单核强在 “选择对的 kernel”。同一 dtype 下它会在 A64/SVE、hybrid/interleaved MMLA kernel 间切换：小 M 或窄 N 更容易选 hybrid，宽 N 或大 K 更容易选 interleaved。这个 selector 让 ACL 在 L2/GT_L2 区间特别稳，I8 单核优势也主要来自这里。

i8gemm SVE 单核强在大规模主循环。GT_L2 BF16 中位数为 235.6 GOPS，略高于 ACL 的 232.1 GOPS；GT_L2 I8 中位数为 416.6 GOPS，接近 ACL 的 466.2 GOPS。大 shape 下 `bfmmla/smmla` compute 占主导，i8gemm SVE 的直接 tiled kernel 能发挥出来。短板是 L1/H2，小规模下 pack A、tail kernel、dispatch 分支和固定 tile 的开销占比偏高。

KleidiAI 单核的固定微核路径很干净，小规模尤其有优势。BF16 L1 中位数 68.7 GOPS，高于 ACL 的 50.0；I8 L1 中位数 133.8 GOPS，高于 ACL 的 68.1。但它没有 ACL 的 shape selector，本次 adapter 也没有 N/K split 或更复杂的 workspace 调度；规模增大后，单一固定 NEON microkernel 的上限和 packed data 复用策略开始落后于 ACL/i8gemm SVE。

i8gemm NEON 的单核定位更像兼容 baseline。I8 L1/H2 还能在少数 shape 上胜出，但 BF16/I8 的总体中位数都明显低于 ACL 和 i8gemm SVE。对支持 SVE BF16/SVE I8MM 的机器，NEON 版本主要用于验证和 fallback，不是优先性能路径。
