# BF16/I8 Dispatch Sync And M8 Benchmark

日期：2026-06-08

## 改动范围

本轮把 SVE BF16 调度层中已经验证过的几项保守优化同步到：

| 路径 | 同步内容 |
| --- | --- |
| BF16 NEON | 线程数 clamp、M/N split 选择策略、N split scratch pool 外置 |
| I8 NEON | 线程数 clamp、M/N split 选择策略、N split scratch pool 外置 |
| I8 SVE | 线程数 clamp、M/N split 选择策略 |

没有直接改动 NEON `.S` 的指令排布。原因是 NEON kernel 的 `A_reorder` 是 kernel 内部 first-N 在线写入、后续 N-block 复用；如果多个 N split 线程共享同一个 `A_reorder` 会发生并发写。因此本轮只把 per-thread `malloc/free` 从 parallel worker 内移到外部一次性分配，每个线程使用固定 slice。

## Correctness

| 测试 | 结果 |
| --- | --- |
| `make -C tests test-sve` | pass，`SVE correctness OK: 6000 wrapper+dispatch cases` |
| `make -C tests test-neon` | pass，NEON BF16/I8 全部 sweep 通过 |

## Production Wrapper Benchmark

命令示例：

```bash
cd tests
./build/bench_perf_sve --mt-both M K N threads
./build/bench_perf --mt-both M K N threads
```

SVE production：

| M | K | N | threads | BF16 GFLOPS | BF16 eff | I8 GOPS | I8 eff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 512 | 512 | 1 | 276.87 | 83.46% | 243.81 | 36.75% |
| 64 | 512 | 512 | 4 | 767.50 | 57.86% | 895.60 | 33.75% |
| 64 | 4096 | 1024 | 1 | 272.69 | 82.20% | 364.31 | 54.92% |
| 64 | 4096 | 1024 | 4 | 777.38 | 58.58% | 1071.54 | 40.38% |
| 512 | 512 | 4096 | 1 | 266.40 | 80.31% | 225.63 | 34.00% |
| 512 | 512 | 4096 | 4 | 772.15 | 58.20% | 704.53 | 26.55% |

NEON production：

| M | K | N | threads | BF16 GFLOPS | BF16 speedup | I8 GOPS | I8 speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 512 | 512 | 1 | 170.6 | 1.00x | 438.7 | 1.00x |
| 64 | 512 | 512 | 4 | 581.8 | 3.40x | 1259.5 | 2.83x |
| 64 | 4096 | 1024 | 1 | 163.0 | 1.00x | 379.4 | 1.00x |
| 64 | 4096 | 1024 | 4 | 542.1 | 3.33x | 1118.3 | 2.99x |
| 512 | 512 | 4096 | 1 | 169.6 | 1.00x | 381.0 | 1.00x |
| 512 | 512 | 4096 | 4 | 588.0 | 3.47x | 1182.6 | 3.04x |

说明：`bench_perf` 里的 Strategy 文案仍按旧启发式打印，实际 dispatch 已使用新的 split 判断。

## M8 Kernel Attribution

命令：

```bash
cd tests
KEEP_CSV=1 WRITE_XLSX=0 RUN_TAILS=0 \
CASES="64,512,512 64,4096,1024 512,512,4096" \
THREADS="1 4" REPS=30 WARMUP=10 RUNS=3 \
OUT=../results/m8/m8_sync_test_parts.csv ./run_m8_parts.sh

KEEP_CSV=1 WRITE_XLSX=0 RUN_TAILS=0 \
CASES="64,4096,1024 512,512,4096" \
THREADS="1" REPS=30 WARMUP=10 RUNS=3 PRUNE_BIG_CASE_THREADS=0 \
OUT=../results/m8/m8_sync_test_parts_1t_big.csv ./run_m8_parts.sh
```

Raw CSV：

- `results/m8/m8_sync_test_parts.csv`
- `results/m8/m8_sync_test_parts_1t_big.csv`

汇总 GFLOPS：

| M | K | N | threads | SVE compute | NEON compute | SVE full | SVE no-store | SVE no-load | SVE f32 | SVE bf16 | SVE bias | NEON f32 | NEON bf16 | NEON bias |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 512 | 512 | 1 | 331.30 | 179.65 | 256.32 | 286.24 | 286.30 | 261.05 | 267.71 | 254.39 | 170.65 | 172.76 | 169.52 |
| 64 | 512 | 512 | 4 | 1012.34 | 618.86 | 776.16 | 837.79 | 915.49 | 766.62 | 769.41 | 751.12 | 591.02 | 595.35 | 590.98 |
| 64 | 4096 | 1024 | 1 | 331.70 | 180.06 | 263.91 | 266.77 | 322.75 | 263.21 | 263.76 | 262.47 | 162.93 | 162.95 | 162.81 |
| 64 | 4096 | 1024 | 4 | 1056.71 | 627.27 | 646.57 | 653.09 | 1036.40 | 649.41 | 647.27 | 646.00 | 512.89 | 473.90 | 467.99 |
| 512 | 512 | 4096 | 1 | 331.67 | 179.73 | 262.66 | 292.09 | 291.03 | 262.83 | 266.10 | 255.27 | 166.47 | 168.83 | 165.25 |
| 512 | 512 | 4096 | 4 | 1058.77 | 627.19 | 664.99 | 694.34 | 962.48 | 663.60 | 660.95 | 659.55 | 564.51 | 564.91 | 555.18 |

## 结论

1. SVE M8 compute-only baseline 稳定接近 331 GFLOPS/thread，说明计算指令本身可以打满；full kernel 的主要差距仍来自 A/B load、store 和内存层级。
2. SVE full 单线程约 78-80%，小 L2 shape 稍高；4 线程 full 对 330×threads 的利用率约 49-59%。
3. NEON M8 compute-only 在这台机器约 180 GFLOPS，full 单线程约 163-173 GFLOPS；4 线程约 468-595 GFLOPS，整体低于 SVE。
4. BF16/I8 NEON 和 I8 SVE 的调度同步属于低风险优化：减少过度开线程和 worker 内 malloc/free，correctness 已覆盖通过。
5. I8 SVE 仍明显受 kernel 实现限制：当前是 C intrinsic + 临时数组 scatter/gather 路径，调度同步无法替代手写 `.S` kernel 的收益。

