# Current Machine M8 Lite Benchmark Analysis

## 1. 测试范围

本报告只分析本机本次生成的结果，不引用之前 80 核日志或其他机器数据。

数据文件：

- `results/m8/current_m8_parts.csv`
- `results/m8/current_m8_stride.csv`
- `results/m8/current_m8_batch.csv`
- `results/m8/current_m8_dispatch.csv`
- `results/m8/current_m8_lite.xlsx`

测试命令核心参数：

```bash
CASE_MODE=lite THREADS="1 2 4 8" REPS=10 WARMUP=3 RUNS=3
RUN_TAILS=0
PART_VARIANTS="full prepacked nostore noload"
STORE_IMPLS="sve neon" STORE_MODES=f32
RUN_STRIDE=1 STRIDE_FACTORS="1 2"
RUN_BATCH=1 BATCH_COUNTS="1 4 16"
RUN_DISPATCH=1 DISPATCH_IMPLS="sve neon" DISPATCH_DTYPES="bf16 i8"
```

由于启用了大 case 剪枝，12 个 lite case 中实际进入统计的是 10 个 shape、27 个 shape/thread 组合。被剪掉的是每线程数据量超过阈值的大矩阵低线程组合。

## 2. Parts 结果

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

SVE full 最差的 compute 利用场景：

| M | K | N | Threads | SVE full | SVE compute-only | full / compute |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 4096 | 1024 | 8 | 999.9 | 1856.8 | 53.9% |
| 16 | 512 | 4096 | 8 | 376.5 | 654.4 | 57.5% |
| 64 | 4096 | 64 | 2 | 383.1 | 654.0 | 58.6% |
| 16 | 512 | 4096 | 2 | 394.4 | 650.7 | 60.6% |
| 64 | 512 | 4096 | 2 | 401.7 | 661.1 | 60.8% |

## 3. 多线程扩展

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

## 4. Stride 影响

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

## 5. 多 GEMM / B 工作集压力

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

## 6. BF16 / I8 Dispatch 对比

dispatch sheet 测的是 public dispatch 路径，不是 M8 attribution 变体。

GFLOPS/GOPS 中位数：

| DType | Impl | T=1 | T=2 | T=4 | T=8 |
|---|---|---:|---:|---:|---:|
| BF16 | SVE | 187.1 | 404.4 | 750.2 | 1330.4 |
| BF16 | NEON | 161.6 | 302.0 | 545.9 | 984.5 |
| I8 | SVE | 267.5 | 478.8 | 899.2 | 1766.7 |
| I8 | NEON | 369.6 | 606.9 | 1145.2 | 2132.3 |

SVE / NEON 中位比值：

| DType | T=1 | T=2 | T=4 | T=8 |
|---|---:|---:|---:|---:|
| BF16 | 115.8% | 118.5% | 131.7% | 122.1% |
| I8 | 75.0% | 81.9% | 78.0% | 82.3% |

结论：

- 本机 BF16 dispatch 上，SVE 中位明显优于 NEON。
- 本机 I8 dispatch 上，SVE 明显落后 NEON，中位只有 NEON 的约 75% 到 82%。
- I8 SVE 仍是后续更值得继续优化的方向；BF16 SVE 当前主要问题不是是否超过 NEON，而是 full path 距离自己的 compute-only 还有明显差距。

## 7. 本机结论

1. BF16 SVE compute-only 很强，但 full path 中位只有 compute-only 的约 62.5%。
2. 本机 BF16 SVE full 的主要损失来自 A/B load 和相关数据路径；store 不是主瓶颈。
3. `lda/ldb/ldc=2x` 对 SVE 影响很小，对 NEON 有更明显影响，最差约 20%。
4. 多 GEMM batch 压力会让 SVE 小/中等 case 掉速，最差约 29%；NEON 在这组数据里更稳。
5. BF16 dispatch：SVE > NEON。I8 dispatch：SVE < NEON。
6. 当前机器 8 线程下，SVE full 的扩展效率低于 compute-only，也低于 NEON f32；下一步应优先处理 SVE full 的 load/reorder/cache 路径。

## 8. 建议

- BF16 SVE：继续围绕 AB load 调度、A reorder 数据路径、B panel 复用做优化；store 优化优先级较低。
- NEON：如果 padded stride 是真实业务场景，优先看 NEON strided store/load 地址路径，当前 stride2 的损失比 SVE 明显。
- 多 GEMM：避免简单 round-robin 跑大量独立 B panel。更合适的是按 B 或 shape 分组，让同一 B panel 的计算尽量连续完成。
- I8 SVE：需要单独优化 kernel/dispatch；当前 dispatch 中位明显落后 NEON，不应直接沿用 BF16 SVE 的判断。
