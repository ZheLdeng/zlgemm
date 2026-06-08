# Current Machine M8 Lite Benchmark Analysis

## 1. 测试范围

本报告只分析本机本次生成的数据，不引用之前 80 核日志或其他机器数据。

最终测试输出：

- `results/m8/current_m8_lite.xlsx`
- `results/m8/current_m8_i8_dispatch_batch.xlsx`

脚本默认 `KEEP_CSV=0`，中间 CSV 只放在临时目录，最终只保留 xlsx。也就是说一次测试默认只输出一个结果文件。

BF16 M8 lite 测试核心参数：

```bash
CASE_MODE=lite THREADS="1 2 4 8" REPS=10 WARMUP=3 RUNS=3
RUN_TAILS=0
PART_VARIANTS="full prepacked nostore noload"
STORE_IMPLS="sve neon" STORE_MODES=f32
RUN_STRIDE=1 STRIDE_FACTORS="1 2"
RUN_BATCH=1 BATCH_COUNTS="1 4 16"
RUN_DISPATCH=1 DISPATCH_IMPLS="sve neon" DISPATCH_DTYPES="bf16 i8"
```

I8 batch 专项测试核心参数：

```bash
CASE_MODE=lite THREADS="1 2 4 8" REPS=10 WARMUP=3 RUNS=3
RUN_PARTS=0 RUN_BASELINES=0 RUN_STORES=0 RUN_TAILS=0
RUN_STRIDE=0 RUN_BATCH=0
RUN_DISPATCH=1 DISPATCH_IMPLS="sve neon" DISPATCH_DTYPES=i8
DISPATCH_BATCH_COUNTS="1 4 16"
```

由于启用了大 case 剪枝，12 个 lite case 中实际进入统计的是 10 个 shape、27 个 shape/thread 组合。被剪掉的是每线程数据量超过阈值的大矩阵低线程组合。

## 2. BF16 Parts 结果

各 variant 的 GFLOPS 中位数：

| Variant | T=1 | T=2 | T=4 | T=8 |
|---|---:|---:|---:|---:|
| `sve_compute_only` | 331.5 | 654.0 | 1058.6 | 1829.0 |
| `sve_full` | 266.0 | 401.7 | 647.0 | 1215.0 |
| `sve_prepacked` | 276.3 | 446.9 | 652.8 | 1217.4 |
| `sve_noload` | 279.0 | 567.3 | 962.3 | 1699.7 |
| `sve_nostore` | 294.1 | 426.7 | 653.1 | 1223.7 |
| `neon_compute_only` | 165.6 | 326.5 | 659.8 | 1272.3 |
| `neon_f32` | 169.0 | 303.8 | 508.7 | 908.4 |

SVE full 相对各拆分项的中位关系：

| Threads | full / compute-only | prepacked / full | noload / full | nostore / full | SVE f32 / NEON f32 |
|---:|---:|---:|---:|---:|---:|
| 1 | 80.2% | 108.3% | 108.0% | 110.6% | 154.8% |
| 2 | 60.8% | 102.0% | 144.9% | 106.2% | 131.8% |
| 4 | 62.4% | 100.9% | 145.4% | 103.6% | 129.6% |
| 8 | 66.7% | 102.7% | 140.0% | 101.6% | 130.1% |

结论：

- 本机 SVE compute-only 明显强于 NEON compute-only，中位约 `1.69x`。
- SVE full 没有把 compute-only 的能力完全转成端到端性能；全局中位 `full / compute-only` 约 `62.5%`。
- `noload / full` 在 2/4/8 线程下约 `1.40x ~ 1.45x`，说明主要损失仍在 A/B load 和相关数据路径。
- `nostore / full` 只有约 `1.02x ~ 1.06x`，store 在这组本机 lite 数据中不是首要瓶颈。
- `prepacked / full` 大多只有约 `1.0x ~ 1.03x`，说明这组测试里 A reorder 不是独立最大项，或者已被其他 load/compute 开销掩盖。

## 3. BF16 最差 compute 利用场景

| M | K | N | Threads | SVE full | SVE compute-only | full / compute |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 4096 | 1024 | 8 | 999.9 | 1856.8 | 53.9% |
| 16 | 512 | 4096 | 8 | 376.5 | 654.4 | 57.5% |
| 64 | 4096 | 64 | 2 | 383.1 | 654.0 | 58.6% |
| 16 | 512 | 4096 | 2 | 394.4 | 650.7 | 60.6% |
| 64 | 512 | 4096 | 2 | 401.7 | 661.1 | 60.8% |

这些 case 差的原因：

| M | K | N | Threads | 原因 |
|---:|---:|---:|---:|---|
| 64 | 4096 | 1024 | 8 | `noload=1806.3` 已接近 `compute-only=1856.8`，但 `nostore=1009.9` 只比 full 略高，说明主要瓶颈是 A/B load 和 B panel 数据路径；store 不是主因。`prepacked=712.8` 低于 full，说明这组不是 A reorder 单项问题。 |
| 16 | 512 | 4096 | 8 | `noload=562.3` 明显高于 full `376.5`，load 是主因；`nostore=418.2` 有小幅提升，store 有影响但不是第一瓶颈。小 M 大 N 下线程拆分和 B 流量更容易放大损失。 |
| 64 | 4096 | 64 | 2 | `noload=564.4` 高于 full `383.1`，load 是主因；`prepacked=446.9` 也有提升，说明 K 大 N 小时第一轮 A reorder/cached 路径也会贡献一部分损失。 |
| 16 | 512 | 4096 | 2 | `noload=580.4` 明显高于 full `394.4`，但 `nostore=403.1` 几乎等于 full，说明 2 线程下这类小 M 大 N 仍然主要卡在 AB load 和 B 数据路径。 |
| 64 | 512 | 4096 | 2 | `noload=582.2` 高于 full `401.7`，`nostore=426.7` 只有小幅收益；N 很大时 B 流量和每线程工作粒度比 store 更关键。 |

归纳：这些最差 case 的共同点不是 zero/store，而是 `K*N` 或 `N` 足够大时的 B 访问压力，以及小 M 或高线程时每个线程可复用的数据不足。

另外，`M=128/512,K=4096,N=1024,T=4` 这组也有类似特征：`noload` 基本接近 compute-only，但 `nostore` 几乎等于 full，所以大 K 大 N 场景的主要问题仍是 load/B panel，而不是 store。

## 4. BF16 多线程扩展

只看同一个 shape 里有 T=1 基线的数据，中位扩展效率如下：

| Variant | T=2 | T=4 | T=8 |
|---|---:|---:|---:|
| `sve_compute_only` | 98.6% | 82.5% | 74.0% |
| `sve_full` | 81.4% | 73.6% | 64.3% |
| `neon_f32` | 89.9% | 82.9% | 76.0% |

结论：

- SVE compute-only 在 2 线程接近线性，4/8 线程开始下降。
- SVE full 的扩展比 compute-only 更差，说明除了线程调度，还有 load/reorder/cache 相关损失。
- NEON f32 的 8 线程扩展效率在这组样本里比 SVE full 更稳。

## 5. BF16 Stride 影响

stride 测试使用：

- `stride_factor=1`: `lda=K, ldb=K, ldc=N`
- `stride_factor=2`: `lda=2K, ldb=2K, ldc=2N`

`stride_factor=2` 相对 `stride_factor=1` 的中位性能：

| Impl | T=1 | T=2 | T=4 | T=8 |
|---|---:|---:|---:|---:|
| SVE | 98.0% | 100.7% | 99.6% | 99.4% |
| NEON | 95.8% | 97.4% | 95.5% | 96.8% |

最差 stride case：

| M | K | N | Threads | Impl | stride1 | stride2 | stride2 / stride1 |
|---:|---:|---:|---:|---|---:|---:|---:|
| 512 | 512 | 4096 | 4 | NEON | 567.8 | 451.8 | 79.6% |
| 512 | 512 | 4096 | 8 | NEON | 1048.7 | 882.4 | 84.1% |
| 64 | 512 | 4096 | 8 | NEON | 1036.1 | 938.8 | 90.6% |
| 64 | 4096 | 64 | 4 | SVE | 729.3 | 682.0 | 93.5% |

结论：

- SVE 对 `lda/ldb/ldc=2x` 基本不敏感，中位损失很小。
- NEON 对 stride 更敏感，尤其 `M=512,K=512,N=4096` 的多线程 case，最大损失约 20%。
- 如果目标场景大量使用 padded leading dimension，当前本机数据更支持优先优化 NEON strided 路径。

## 6. BF16 多 GEMM / B 工作集压力

batch 测试保持单个 GEMM 的 M/K/N 不变，轮流跑多组独立 `A/B/C/A_reorder`，用 `batch_count` 放大活跃工作集。

`batch_count=4/16` 相对 `batch_count=1` 的中位性能：

| Impl | Threads | batch4 / batch1 | batch16 / batch1 |
|---|---:|---:|---:|
| SVE | 1 | 94.6% | 90.4% |
| SVE | 2 | 91.7% | 86.5% |
| SVE | 4 | 96.5% | 95.3% |
| SVE | 8 | 96.7% | 96.5% |
| NEON | 1 | 98.3% | 96.1% |
| NEON | 2 | 98.4% | 94.2% |
| NEON | 4 | 98.0% | 92.6% |
| NEON | 8 | 97.8% | 96.8% |

最差 batch16 case：

| M | K | N | Threads | Impl | batch1 | batch16 | batch16 / batch1 |
|---:|---:|---:|---:|---|---:|---:|---:|
| 16 | 512 | 128 | 8 | SVE | 440.0 | 312.5 | 71.0% |
| 16 | 512 | 128 | 2 | SVE | 435.9 | 311.2 | 71.4% |
| 16 | 512 | 128 | 4 | SVE | 445.8 | 323.2 | 72.5% |
| 64 | 4096 | 64 | 4 | SVE | 702.5 | 546.0 | 77.7% |
| 64 | 512 | 512 | 4 | SVE | 810.6 | 641.0 | 79.1% |

结论：

- batch 放大后没有出现全局崩盘，但 SVE 对小/中等工作集的 batch 压力更敏感。
- `M=16,K=512,N=128` 在 batch16 下掉到约 71%，这类小 GEMM 同时存在很多组 B 时，cache residency 和调度开销都会放大。
- NEON 的 batch 中位更稳，batch16 大多仍在 92% 以上。
- 如果业务是多个小 GEMM 并发，建议按 B panel 分组执行，避免在太多独立 B 之间频繁轮转。

## 7. I8 Dispatch 结果

I8 测的是 public dispatch 路径，不是 M8 attribution 变体。这里不和 BF16 做对比，只看 I8 自己在不同实现、线程数、batch 压力下的表现。

I8 GOPS 中位数：

| Impl | T=1 | T=2 | T=4 | T=8 |
|---|---:|---:|---:|---:|
| SVE | 270.0 | 471.5 | 893.9 | 1737.8 |
| NEON | 373.3 | 614.5 | 1145.1 | 2145.5 |

I8 多线程扩展效率：

| Impl | T=2 | T=4 | T=8 |
|---|---:|---:|---:|
| SVE | 84.5% | 72.7% | 65.8% |
| NEON | 83.1% | 73.2% | 69.7% |

结论：

- I8 dispatch 在 2 线程后扩展效率开始下降，4/8 线程下降更明显。
- SVE 和 NEON 的扩展趋势类似，说明 I8 的多线程损失不只是某一个 ISA 路径的问题，也和 shape 粒度、B/C 工作集、OpenMP 调度有关。
- 当前这轮没有做 I8 `lda/ldb/ldc=2x` 的有效测试：public dispatch API 不暴露 `lda/ldb/ldc`，直接用 dispatch 结果不能代表低层 stride 成本。I8 stride 需要单独的 low-level kernel bench，避免把 padding copy 或 pack 成本误当成 kernel stride 成本。

## 8. I8 多 GEMM / B 工作集压力

`batch_count=4/16` 相对 `batch_count=1` 的中位性能：

| Impl | Threads | batch4 / batch1 | batch16 / batch1 |
|---|---:|---:|---:|
| SVE | 1 | 95.0% | 94.7% |
| SVE | 2 | 97.8% | 94.4% |
| SVE | 4 | 98.2% | 95.1% |
| SVE | 8 | 98.0% | 96.1% |
| NEON | 1 | 95.0% | 89.9% |
| NEON | 2 | 96.4% | 90.6% |
| NEON | 4 | 96.6% | 91.2% |
| NEON | 8 | 95.6% | 93.2% |

最差 batch16 case：

| M | K | N | Threads | Impl | batch1 | batch16 | batch16 / batch1 |
|---:|---:|---:|---:|---|---:|---:|---:|
| 16 | 512 | 128 | 2 | SVE | 454.0 | 366.2 | 80.7% |
| 16 | 512 | 128 | 8 | SVE | 455.3 | 370.7 | 81.4% |
| 64 | 4096 | 64 | 8 | NEON | 2113.9 | 1721.3 | 81.4% |
| 16 | 512 | 128 | 4 | SVE | 452.1 | 370.2 | 81.9% |
| 64 | 512 | 512 | 8 | NEON | 2219.6 | 1829.7 | 82.4% |

结论：

- I8 batch16 的中位损失比 BF16 SVE batch 最差情况更温和，但小 GEMM 和 `K` 很大的窄 N case 仍会明显掉速。
- `M=16,K=512,N=128` 是典型小工作量 case：多线程下 kernel 时间短，batch 轮转会放大调度和 cache 切换成本。
- `M=64,K=4096,N=64` 是 K 很大、N 很窄的 case：B panel 和 A 数据流量都高，但单个 N 方向可并行度有限，batch16 会降低局部性。

## 9. 本机结论

1. BF16 SVE compute-only 很强，但 full path 中位只有 compute-only 的约 62.5%。
2. 本机 BF16 SVE full 的主要损失来自 A/B load 和相关数据路径；store 不是主瓶颈。
3. BF16 `lda/ldb/ldc=2x` 对 SVE 影响很小，对 NEON 有更明显影响，最差约 20%。
4. BF16 多 GEMM batch 压力会让 SVE 小/中等 case 掉速，最差约 29%；NEON 在这组数据里更稳。
5. I8 dispatch 的扩展在 4/8 线程也有明显下降；batch16 对小 GEMM 和 K 大窄 N case 影响最大。
6. 当前机器 8 线程下，BF16 SVE full 的扩展效率低于 compute-only；下一步应优先处理 SVE full 的 load/reorder/cache 路径。

## 10. 建议

- BF16 SVE：继续围绕 AB load 调度、A reorder 数据路径、B panel 复用做优化；store 优化优先级较低。
- NEON：如果 padded stride 是真实业务场景，优先看 NEON strided store/load 地址路径，当前 stride2 的损失比 SVE 明显。
- 多 GEMM：避免简单 round-robin 跑大量独立 B panel。更合适的是按 B 或 shape 分组，让同一 B panel 的计算尽量连续完成。
- I8：补一个 low-level i8 stride bench，真实传入 `lda/ldb/ldc=2x`，再判断 I8 stride 是否和 BF16 一样不敏感。
