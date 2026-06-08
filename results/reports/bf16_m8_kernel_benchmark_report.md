# BF16 M8 Kernel Benchmark Report

日期：2026-06-08

对象：`bf16gemm_sve_m8_nld.S` 和 `bf16gemm_neon_m8_nld.S` 这一组 standalone M8 kernel，以及完整 full-tail 路径中的 SVE/NEON 对照。

## 1. 结果文件整理

根目录里原来有多份临时 CSV/XLSX：

- `m8_fulltail_sve.xlsx` / `m8_fulltail_neon.xlsx`：单独 SVE/NEON workbook，内容已经被合并 workbook 覆盖。
- `m8_fulltail_sve_parts.csv` / `m8_fulltail_neon_parts.csv` / `m8_parts_results.csv`：只有表头，属于无效临时输出。
- `m8_fulltail_sve.log` / `m8_fulltail_neon.log`：只记录极少构建信息。

现在保留的有效数据集中在：

| 文件 | 用途 |
| --- | --- |
| `results/m8/m8_parts_sve_neon_representative.csv` | 新跑的代表性 M8 parts + SVE/NEON 对照 |
| `results/m8/m8_fulltail_sve_tail.csv` | SVE full-tail 原始结果 |
| `results/m8/m8_fulltail_neon_tail.csv` | NEON full-tail 原始结果 |
| `results/m8/m8_fulltail_all.xlsx` | 合并后的 workbook |
| `results/m8/archive/` | 归档的旧临时结果 |

根目录现在只保留非 M8 临时结果：`shape.csv` 和 `test_correctness.log`。

## 2. 测试口径

峰值口径：

- SVE 单核 BF16 峰值仍按 `330 GFLOPS` 估算。
- 多线程峰值按 `330 * threads` 估算。
- 当前机器 `nproc=8`，所以报告里的多线程结论主要覆盖 `1/8` 线程。

parts 口径：

- `sve_compute_only` / `neon_compute_only`：去掉 load/store/zero，只保留计算和循环控制。
- `sve_full` / `sve_nozero` / `sve_nostore` / `sve_noload`：SVE 归因变体，用于估计 zero/store/AB-load 的影响。
- `sve_f32/neon_f32`、`sve_bf16/neon_bf16`、`sve_bias/neon_bias`：standalone full-output 对比。NEON 当前没有完整的 nozero/nostore/noload 归因变体。

影响估算：

```text
impact = 1 - full_GFLOPS / removed_part_GFLOPS
```

这些影响不能直接相加，因为 load/store/compute 之间有重叠。

## 3. M8 Parts 瓶颈拆分

下面使用 `results/m8/m8_parts_sve_neon_representative.csv` 中的代表 shape。

| M | K | N | T | SVE full | SVE compute | full util | compute share | zero | store | AB load | NEON f32 | SVE/NEON |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 512 | 4096 | 1 | 148.1 | 190.1 | 44.9% | 77.9% | -0.1% | -38.2% | 5.1% | 89.7 | 1.65x |
| 16 | 512 | 4096 | 8 | 85.8 | 647.4 | 3.2% | 13.2% | -2.2% | -0.5% | 85.0% | 289.9 | 0.30x |
| 64 | 512 | 4096 | 1 | 133.3 | 177.5 | 40.4% | 75.1% | 2.6% | 14.7% | 14.5% | 89.7 | 1.49x |
| 64 | 512 | 4096 | 8 | 433.8 | 1807.6 | 16.4% | 24.0% | 64.7% | -71.6% | 74.2% | 191.0 | 2.27x |
| 512 | 4096 | 1024 | 1 | 263.4 | 331.7 | 79.8% | 79.4% | -0.1% | 1.0% | 18.1% | 162.4 | 1.62x |
| 512 | 4096 | 1024 | 8 | 1223.6 | 1850.7 | 46.3% | 66.1% | 0.0% | 0.5% | 32.2% | 888.8 | 1.38x |
| 2048 | 512 | 4096 | 1 | 132.0 | 165.3 | 40.0% | 79.9% | 0.6% | 10.0% | 9.1% | 163.6 | 0.81x |
| 2048 | 512 | 4096 | 8 | 1248.7 | 1851.7 | 47.3% | 67.4% | 0.4% | 4.3% | 27.2% | 1047.2 | 1.19x |
| 2048 | 4096 | 64 | 1 | 238.5 | 331.6 | 72.3% | 71.9% | 1.0% | -2.7% | 14.3% | 153.9 | 1.55x |
| 2048 | 4096 | 64 | 8 | 1232.1 | 1851.0 | 46.7% | 66.6% | 0.2% | 2.3% | 19.3% | 949.1 | 1.30x |

结论：

1. SVE compute-only 在大块单线程场景能到 `331 GFLOPS` 左右，说明计算核心本身可以接近单核峰值。
2. 多线程 compute-only 约 `1800..1850 GFLOPS`，相当于 8 核峰值的 `68%..70%`。所以 full kernel 多线程上限首先被 compute-only 多线程效率限制。
3. 对大块 M8 standalone，SVE full 相比 NEON f32 通常更快，典型 `1.2x..1.6x`。但是 `M=2048,K=512,N=4096,T=1` 这类大 B 流场景里，NEON f32 反而更高，说明 SVE 当前数据流对某些 cache/带宽场景不够稳。
4. 小 M 多线程依然很差，`M=16,T=8` 的 SVE full 几乎不可用。这类 shape 不应该强拆到很多线程。
5. `zero` 的影响整体不是主瓶颈。大块场景中更值得关注的是 AB load，8 线程下 AB-load impact 常见 `19%..32%`。

## 4. SVE/NEON 输出路径对比

| M | K | N | T | SVE f32 | SVE bf16 | SVE bias | NEON f32 | NEON bf16 | NEON bias |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 512 | 4096 | 1 | 111.0 | 148.5 | 145.6 | 89.7 | 84.2 | 89.3 |
| 16 | 512 | 4096 | 8 | 363.7 | 358.9 | 105.1 | 289.9 | 83.9 | 299.3 |
| 64 | 512 | 4096 | 1 | 133.0 | 133.9 | 131.0 | 89.7 | 85.5 | 83.7 |
| 64 | 512 | 4096 | 8 | 1227.5 | 252.6 | 251.8 | 191.0 | 280.6 | 184.6 |
| 512 | 4096 | 1024 | 1 | 263.2 | 263.1 | 262.0 | 162.4 | 162.7 | 162.8 |
| 512 | 4096 | 1024 | 8 | 1222.8 | 1223.0 | 1221.1 | 888.8 | 896.0 | 879.8 |
| 2048 | 512 | 4096 | 1 | 131.4 | 133.6 | 129.1 | 163.6 | 161.6 | 164.9 |
| 2048 | 512 | 4096 | 8 | 1248.8 | 1244.4 | 1237.9 | 1047.2 | 1053.8 | 1033.8 |
| 2048 | 4096 | 64 | 1 | 235.8 | 232.8 | 234.9 | 153.9 | 154.9 | 153.9 |
| 2048 | 4096 | 64 | 8 | 1248.9 | 1226.8 | 1245.8 | 949.1 | 949.4 | 941.0 |

解释：

- 大块场景里，SVE f32/bf16/bias 三条路径已经很接近，说明 bias 初始化到 accumulator 的方向是对的。
- `M=64,K=512,N=4096,T=8` 的 SVE bf16/bias 明显异常偏低，应该作为后续专项检查点；它不像大块场景的稳定行为。
- NEON 输出三条路径整体较稳定，但绝对性能通常低于 SVE，大块 8 线程下约 `880..1050 GFLOPS`。

## 5. Full-Tail 对比

tail 数据来自 `results/m8/m8_fulltail_sve_tail.csv` 和 `results/m8/m8_fulltail_neon_tail.csv`。

### SVE tail 平均影响

| threads | +1 avg drop | +2 avg drop | +4 avg drop | base avg GFLOPS |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 6.7% | 3.6% | -3.4% | 161.2 |
| 2 | 5.2% | 2.7% | -1.4% | 239.4 |
| 4 | 2.7% | 0.8% | -11.8% | 373.0 |
| 8 | 1.6% | -0.7% | -9.2% | 565.5 |

### NEON tail 平均影响

| threads | +1 avg drop | +2 avg drop | +4 avg drop | base avg GFLOPS |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.7% | 1.9% | -0.7% | 157.9 |
| 2 | 3.6% | 1.5% | -1.5% | 275.5 |
| 4 | 3.7% | 1.4% | -1.6% | 470.7 |
| 8 | 4.5% | 3.3% | 1.3% | 791.1 |

结论：

- SVE 的 `+1` tail 单线程平均损失比 NEON 更高，`6.7%` vs `4.7%`。
- NEON tail 在 8 线程下更稳定，`+1/+2/+4` 都是正 drop 且幅度较小。
- SVE 的 `+4` tail 经常出现负 drop，这通常不是 tail 更快，而是总 M 增大后固定开销被摊薄。看 tail 是否真的好，优先看 `+1/+2`。

## 6. Full-Tail Cache 档位聚合

| impl | cache | T1 base avg | T8 base avg | speedup |
| --- | --- | ---: | ---: | ---: |
| SVE | L1 | 70.1 | 83.5 | 1.19x |
| SVE | H2 | 149.4 | 431.7 | 2.89x |
| SVE | L2 | 210.8 | 836.7 | 3.97x |
| SVE | GT_L2 | 224.3 | 994.2 | 4.43x |
| NEON | L1 | 130.8 | 240.7 | 1.84x |
| NEON | H2 | 168.4 | 842.5 | 5.00x |
| NEON | L2 | 162.5 | 1024.4 | 6.30x |
| NEON | GT_L2 | 162.2 | 1035.7 | 6.39x |

这里是完整 full-tail 路径，不是 standalone M8 parts。趋势很清楚：

- SVE 单线程在 L2/GT_L2 的平均 base 比 NEON 高。
- NEON 多线程线性度明显更稳，尤其 L2/GT_L2 的 8 线程 speedup 到 `6.3x..6.4x`。
- SVE 多线程目前主要输在线性度，不是单核 kernel 算力。

## 7. run_m8_parts.sh 现在包含哪些 NEON 数据

默认已经会跑 NEON：

```bash
STORE_IMPLS="sve neon"
BASELINE_IMPLS="sve neon"
TAIL_IMPLS="sve neon"
```

因此默认输出包括：

- `neon_compute_only`
- `neon_f32`
- `neon_bf16`
- `neon_bias`
- tail sheet 中 `impl=neon`

目前还没有 NEON 的 `nozero/nostore/noload` 完整归因变体；这些细分仍主要用于 SVE。

常用命令：

```bash
# 完整 SVE+NEON parts/store/tail
THREADS=auto c=0-79 RESULTS_XLSX=m8_80c_shape_sweep.xlsx ./run_m8_parts.sh

# 只跑 SVE/NEON full/bf16/bias/compute/tail，不跑 SVE 细分 parts
RUN_PARTS=0 THREADS=auto c=0-79 RESULTS_XLSX=m8_sve_neon_compare.xlsx ./run_m8_parts.sh
```

## 8. 当前结论

1. SVE M8 的 compute-only 单核很强，大块 case 接近理论峰值。
2. SVE full 大块单线程通常比 NEON 快，但多线程线性度不如 NEON。
3. SVE 多线程主要问题是 AB load/cache 行为和每线程任务粒度；不是 zero ACC。
4. NEON 作为对照很有价值：它单核峰值低，但完整 full-tail 多线程更稳，说明 SVE 后续优化应该更多看调度和数据复用，而不是只盯单核计算指令。
5. 小 M 多线程仍要谨慎，尤其 `M=16` 这种场景不应该盲目拆到 8 线程或更多。
