# Current M8 Lite 80c Benchmark Analysis

## 1. 测试范围

本报告只分析 `results/m8/current_m8_lite_80c.xlsx`。

这次 xlsx 里包含两个 sheet：

- `parts`: M8 kernel 拆分项测试，共 1620 条记录。
- `tail`: full/tail M 对比测试，共 162 条记录。

对应命令形态：

```bash
CASE_MODE=lite THREADS=auto c=80-159 RESULTS_XLSX=../results/m8/current_m8_lite_80c.xlsx ./run_m8_parts.sh
```

本次文件没有 `stride`、`batch`、`dispatch` sheet，所以不分析 stride、多 GEMM batch、I8 dispatch。注意：这是这份 xlsx 生成时的状态；脚本已调整为 `CASE_MODE=lite` 默认打开 `RUN_DISPATCH=1`，后续重跑同样命令会多出 `dispatch` sheet，用来观察当前完整 BF16/I8 public dispatcher 的最优路径。

一个关键前提：这份报告里的 `parts` 不是完整对外 dispatcher 的测试，而是裸 M8 kernel attribution benchmark。`bench_m8_parts.c` 的多线程只按 M 方向切分，工作单元是 `M/8` 个 M8 block，不切 N。因此当线程数大于 `M/8`，或者 `M/8` 不能较均匀分给线程时，即使 compute-only 也会掉速。

完整 SVE BF16 对外接口在 `lib/bf16gemm_sve.c` 里仍然有按 N 切的策略：`bf16_use_n_split()` 会在 `M/8 < threads`、或者 B panel 足够大且 `N >= 2M` 时选择 N split，也可以通过 `BF16_SVE_SPLIT=n` 强制指定。本报告不能直接用来否定 N split，只能说明当前裸 M8 parts benchmark 的 M-only 切分限制。

关于“是否基于当前最优代码”：

- `parts` sheet 不是完整最优 dispatcher。它使用 `tests/bench_m8_parts.c` 加 `tests/bf16gemm_sve_m8_nld.S` / `tests/bf16gemm_neon_m8_nld.S`，并由脚本生成 `full/prepacked/noload/nostore/compute_only` 等 attribution 版本。这部分用于分析 M8 kernel 本身和拆分项开销。
- `tail` sheet 是完整对外 dispatcher 路径。它使用 `tests/bench_full_tail.c`，SVE 编译链接 `lib/bf16gemm_sve.c` + `lib/bf16gemm_sve.S`，因此会走当前集成的调度逻辑，包括 M12/M8、tail 和 N split 自动策略。
- 这份旧 xlsx 生成时 `RUN_DISPATCH=0`，所以没有对所有 lite shape 跑完整 public dispatch sheet。后续重跑同样 `CASE_MODE=lite` 命令时，脚本会默认生成 `dispatch` sheet；该 sheet 才是判断“当前最优完整接口”在所有 lite shape 上性能的主要依据。

## 2. BF16 Parts 结果

关键 variant 的 GFLOPS 中位数：

| Variant | T=1 | T=8 | T=16 | T=32 | T=64 | T=80 |
|---|---:|---:|---:|---:|---:|---:|
| `sve_compute_only` | 88.5 | 698.9 | 719.0 | 1033.8 | 990.9 | 977.4 |
| `sve_full` | 77.8 | 607.4 | 614.4 | 904.2 | 865.7 | 853.2 |
| `sve_prepacked` | 89.0 | 705.7 | 718.3 | 1057.9 | 1000.0 | 986.8 |
| `sve_noload` | 87.7 | 693.0 | 719.9 | 1070.3 | 1030.6 | 1014.2 |
| `sve_nostore` | 78.7 | 617.2 | 618.4 | 908.4 | 870.5 | 860.0 |
| `neon_compute_only` | 86.6 | 666.3 | 676.5 | 1003.1 | 969.1 | 956.8 |
| `neon_f32` | 82.2 | 642.4 | 648.4 | 955.6 | 906.0 | 893.7 |

SVE full 相对拆分项的中位关系：

| Threads | full / compute-only | prepacked / full | noload / full | nostore / full | nozero / full |
|---:|---:|---:|---:|---:|---:|
| 1 | 87.5% | 115.1% | 112.5% | 101.7% | 100.8% |
| 8 | 87.3% | 116.6% | 115.2% | 101.6% | 100.1% |
| 32 | 85.9% | 115.8% | 116.7% | 101.6% | 100.2% |
| 64 | 87.0% | 115.0% | 116.4% | 102.4% | 100.1% |
| 80 | 86.8% | 114.6% | 116.3% | 101.6% | 100.0% |

结论：

- SVE full 大约是 compute-only 的 `86% ~ 88%`，比之前 8 线程小范围测试稳定很多。
- `prepacked / full` 约 `1.15x`，说明 A reorder 或 full path 中与 A 数据准备相关的成本比较稳定。
- `noload / full` 约 `1.13x ~ 1.17x`，说明 A/B load 和相关数据路径仍然是主要损失项。
- `nostore / full` 只有约 `1.02x`，store 不是这次 80c 数据里的主瓶颈。
- `nozero / full` 基本等于 `1.00x`，zero ACC 开销可以忽略。

## 3. M Block 数对 Parts 多线程的影响

以下表格用 `sve_full / (threads * 88.5)` 估算每核利用率，其中 `88.5 GFLOPS` 是这次 `sve_compute_only` 单线程中位数。这个指标不是硬件峰值，只用于看当前 benchmark 的多线程有效性。

| M | M8 blocks | T=8 | T=16 | T=32 | T=64 | T=80 |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 2 | 20.4% | 9.9% | 4.8% | 2.2% | 1.7% |
| 64 | 8 | 84.6% | 41.7% | 20.3% | 9.7% | 7.6% |
| 128 | 16 | 88.0% | 87.0% | 42.6% | 20.3% | 16.0% |
| 512 | 64 | 87.5% | 86.3% | 85.0% | 84.0% | 65.6% |
| 2048 | 256 | 86.0% | 85.5% | 85.6% | 83.9% | 66.7% |

原因：

- 当前 parts benchmark 只按 M 切，最多只有 `M/8` 个并行 block。
- `M=16` 只有 2 个 block，所以 T=8/16/32/64/80 都会大量线程空转。
- `M=64` 只有 8 个 block，所以 T=8 附近最好，继续加线程只会降低平均利用率。
- `M=128` 有 16 个 block，所以 T=16 还能保持高利用，T=32 起明显下降。
- `M=512` 有 64 个 block，所以 T=64 仍然有效；T=80 时 64 个 block 不够 80 线程分，性能下降。
- `M=2048` 有 256 个 block，理论上可支持 80 线程，但 `256 / 80 = 3.2`，静态切分时有些线程拿 4 个 block、有些拿 3 个 block，理论负载不均会带来约 `3.2 / 4 = 80%` 的上限；实测 T=80 也明显低于 T=64。

所以这份 parts 数据里 compute-only 也受 block 数影响，不是访存问题，而是裸 M8 benchmark 的 M-only 静态切分粒度问题。完整 dispatcher 的 N split/2D/ngroup/nblock 路径需要看 dispatch 或 public API benchmark，不能从这里直接推出。

## 4. 每个 Shape 的最佳线程数

SVE full 的最佳线程数：

| M | K | N | M8 blocks | Best T | GFLOPS | GFLOPS / thread |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 512 | 128 | 2 | 2 | 143.3 | 71.6 |
| 16 | 512 | 4096 | 2 | 2 | 153.4 | 76.7 |
| 64 | 512 | 512 | 8 | 8 | 591.6 | 74.0 |
| 64 | 512 | 4096 | 8 | 8 | 606.1 | 75.8 |
| 64 | 4096 | 64 | 8 | 10 | 591.0 | 59.1 |
| 64 | 4096 | 1024 | 8 | 8 | 621.7 | 77.7 |
| 128 | 4096 | 1024 | 16 | 16 | 1231.5 | 77.0 |
| 512 | 512 | 4096 | 64 | 64 | 4712.1 | 73.6 |
| 512 | 4096 | 1024 | 64 | 64 | 4807.1 | 75.1 |
| 2048 | 512 | 4096 | 256 | 64 | 4777.1 | 74.6 |
| 2048 | 1024 | 8192 | 256 | 80 | 4779.3 | 59.7 |
| 2048 | 4096 | 64 | 256 | 80 | 4723.9 | 59.0 |

规律：

- 最佳线程数首先受 `M/8` 限制。
- 当 `M/8` 很小，最佳线程数基本等于 `M/8`。
- 当 `M/8` 足够大时，线程数还要考虑整除和负载均衡。`M=512` 有 64 个 block，T=64 比 T=80 更自然；`M=2048` 有 256 个 block，T=64 每线程 4 个 block，分配完全均匀，因此往往比 T=80 更稳。
- `M=2048,N=8192` 和 `M=2048,K=4096,N=64` 的 T=80 绝对 GFLOPS 最高，但每线程性能明显低于 T=64，说明这里是更多线程换吞吐，而不是更高利用率。

## 5. 最差 compute 利用场景

SVE full / compute-only 最低的 case：

| M | K | N | Threads | SVE full | SVE compute-only | full / compute |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 1024 | 8192 | 64 | 4749.3 | 5909.1 | 80.4% |
| 2048 | 1024 | 8192 | 40 | 2744.2 | 3376.4 | 81.3% |
| 512 | 512 | 4096 | 64 | 4712.1 | 5790.5 | 81.4% |
| 2048 | 512 | 4096 | 64 | 4777.1 | 5868.9 | 81.4% |
| 512 | 512 | 4096 | 80 | 4555.4 | 5593.7 | 81.4% |

这些 case 差的原因：

| M | K | N | Threads | 原因 |
|---:|---:|---:|---:|---|
| 2048 | 1024 | 8192 | 64 | `noload=5735.7` 接近 compute-only，`nostore=4840.0` 只比 full 略高；主要损失是 A/B load 和 full path 数据准备，不是 store。 |
| 2048 | 1024 | 8192 | 40 | `prepacked=3089.2`、`noload=3279.6` 都明显高于 full，说明 A reorder 和 AB load 都有稳定开销。 |
| 512 | 512 | 4096 | 64 | `prepacked=5352.3`、`noload=5519.0`、`noload_nostore=5751.3` 逐步接近 compute-only，说明 load/reorder 是主因，store 只贡献小幅损失。 |
| 2048 | 512 | 4096 | 64 | `prepacked=5484.6`、`noload=5598.2` 都明显高于 full，主要仍是 full path 的 A/B 数据路径。 |
| 512 | 512 | 4096 | 80 | M8 blocks 只有 64，小于 80 线程，除了 load/reorder 成本外，还有线程空转和负载粒度问题。 |

归纳：这次 80c 的最差 full/compute 仍然在 80% 以上，主要损失来自 load/reorder 数据路径；但当线程数超过或不适配 `M/8` block 数时，compute-only 本身也会被调度粒度限制。

## 6. Tail 影响

tail sheet 只测了 3 组 base shape：

- `M=512,K=4096,N=1024`
- `M=64,K=512,N=4096`
- `M=2048,K=4096,N=64`

tail_M 分别是 `base_M+1`、`base_M+2`、`base_M+4`。

tail 相对 base 的中位比例：

| Impl | Threads | +1 | +2 | +4 |
|---|---:|---:|---:|---:|
| SVE | 8 | 95.5% | 97.1% | 100.5% |
| SVE | 32 | 97.3% | 98.3% | 101.2% |
| SVE | 64 | 97.7% | 96.7% | 102.1% |
| SVE | 80 | 100.4% | 100.1% | 138.1% |
| NEON | 8 | 99.5% | 99.5% | 99.8% |
| NEON | 32 | 97.9% | 99.3% | 99.7% |
| NEON | 64 | 96.5% | 99.8% | 101.5% |
| NEON | 80 | 97.9% | 99.7% | 100.8% |

最差 tail outlier：

| Impl | Threads | base_M | tail_M | K | N | base | tail | tail/base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SVE | 64 | 512 | 513 | 4096 | 1024 | 5182.4 | 2174.6 | 42.0% |
| SVE | 64 | 512 | 514 | 4096 | 1024 | 5182.4 | 2175.8 | 42.0% |
| SVE | 40 | 512 | 513 | 4096 | 1024 | 2734.7 | 1377.7 | 50.4% |
| SVE | 32 | 512 | 513 | 4096 | 1024 | 2748.8 | 1388.1 | 50.5% |
| SVE | 40 | 512 | 516 | 4096 | 1024 | 2734.7 | 1830.6 | 66.9% |

原因：

- `M=512` 正好是 64 个 M8 block。
- T=64 时 base case 可以做到每线程 1 个完整 block，负载非常均匀。
- `M=513/514` 会额外产生一个很小的 tail block，静态分配下某个线程会多拿一个 block，而这个 tail block 的计算访存比很低，整体 wall time 被这个不均衡放大。
- `M=516` 的 tail block 是 4 行，比 1/2 行好一些，所以损失小于 +1/+2。
- NEON tail 中位更稳，SVE tail 的 outlier 更明显，说明 SVE tail 路径仍值得单独优化或在调度层绕开。

## 7. 结论

1. `current_m8_lite_80c` 的 parts sheet 核心瓶颈不是单纯 kernel 计算能力，而是裸 M8 benchmark 的 M-only 静态切分下的 block 数和负载均衡。
2. 对 parts benchmark 来说，线程数最好不要超过 `M/8`，并且最好让 `M/8` 能比较均匀地分给线程；完整 dispatcher 不受这个单一规则约束，因为它可以选择 N split。
3. `M=64` 这类只有 8 个 M8 block 的 shape，不适合跑 80 线程；compute-only 也会因为线程空转而掉。
4. `M=512` 有 64 个 block，T=64 比 T=80 更合理。
5. `M=2048` 有 256 个 block，T=64 每线程 4 个 block，负载最均匀；T=80 虽然有时绝对 GFLOPS 最高，但每线程利用率明显降低。
6. SVE full 相比 compute-only 的稳定损失主要来自 A reorder 和 AB load 路径；store 和 zero 不是主因。
7. Tail 的平均影响不大，但 `M=512,T=32/40/64,+1/+2` 这类正好打破 block 均衡的场景会出现严重 outlier。

## 8. 优化建议

- benchmark 层：给 `bench_m8_parts.c` 也增加 N 方向切分或 2D 切分，或者补一个专门的 public dispatcher attribution 测试；否则 M 小时无法用 parts sheet 评价完整接口的 80 核真实能力。
- 调度层：线程数可以按 `min(request_threads, M/8)` 先做硬限制；进一步优先选择能整除或接近整除 `M/8` 的线程数。
- tail 层：对 `M % 8 != 0` 的场景，避免让一个线程额外承担低效 tail block；可以考虑 tail 合并到相邻 full block、单独串行尾部、或按 N 方向拆 tail。
- kernel 层：继续优先优化 A reorder 和 AB load 调度；zero 和 store 优先级较低。
