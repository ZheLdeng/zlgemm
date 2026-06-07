# i8gemm 文件夹使用说明

本文档说明当前目录里主要源码的职责，分为两部分：

1. 对外 API、最优多线程调度，以及对应的 `.h/.c/.S`
2. 两套完整测试框架，以及它们涉及的 `.c/.S`

## 1. 对外 API 和多线程调度实现

### BF16 SVE 推荐入口

当前 BF16 SVE 推荐使用专门头文件：

| 文件 | 作用 |
| --- | --- |
| `bf16gemm_sve.h` | SVE BF16 对外头文件，包含最优调度配置 API 和 SVE 命名接口 |
| `bf16gemm.h` | BF16 通用兼容头文件，保留 `bf16gemm_mt_*` 旧接口 |
| `bf16gemm_sve.c` | SVE BF16 多线程调度、A reorder、B pack、padding wrapper |
| `bf16gemm_sve.S` | SVE BF16 完整 kernel，支持 M8 主体和 M4/M2/M1 tail |
| `gemm_params.h` | C 和汇编共享的参数结构 |

推荐外部代码包含：

```c
#include "bf16gemm_sve.h"
```

推荐链接：

```bash
cc -O3 -fopenmp -mcpu=native your_code.c bf16gemm_sve.c bf16gemm_sve.S -lm
```

如果需要显式架构参数，可以替换成：

```bash
cc -O3 -fopenmp -march=armv8.6-a+sve+bf16 your_code.c bf16gemm_sve.c bf16gemm_sve.S -lm
```

### BF16 SVE 对外函数

`bf16gemm_sve.h` 提供 SVE 命名接口：

| 函数 | 输出 | 说明 |
| --- | --- | --- |
| `bf16gemm_sve_f32` | fp32 | wrapper：内部完成 padding、B pack、dispatch |
| `bf16gemm_sve_bias_f32` | fp32 | wrapper：计算 `A*B + bias`，bias 为 fp32，长度 N |
| `bf16gemm_sve_bf16` | bf16 | wrapper：最终输出 cast 到 bf16 |
| `bf16gemm_sve_pack_B` | packed B | caller 自己管理 B pack 时使用 |
| `bf16gemm_sve_dispatch_f32` | fp32 | zero-allocation dispatch，要求输入已经 pad/pack |
| `bf16gemm_sve_dispatch_bias_f32` | fp32 | zero-allocation bias dispatch，要求 bias 已 pad 到 N_r |
| `bf16gemm_sve_dispatch_bf16` | bf16 | zero-allocation bf16 output dispatch |

通用兼容接口仍然存在：

| 函数 | 对应 SVE 命名接口 |
| --- | --- |
| `bf16gemm_mt` | `bf16gemm_sve_f32` |
| `bf16gemm_mt_bias_f` | `bf16gemm_sve_bias_f32` |
| `bf16gemm_mt_nld_b` | `bf16gemm_sve_bf16` |
| `bf16gemm_mt_dispatch` | `bf16gemm_sve_dispatch_f32` |
| `bf16gemm_mt_dispatch_bias_f` | `bf16gemm_sve_dispatch_bias_f32` |
| `bf16gemm_mt_dispatch_nld_b` | `bf16gemm_sve_dispatch_bf16` |

### 当前最优调度策略

当前默认调度策略已经集成在 `bf16gemm_sve.c`，并通过 `bf16gemm_sve.h` 暴露：

| 策略项 | 默认值 | 目的 |
| --- | ---: | --- |
| `split_policy` | `BF16GEMM_SVE_SPLIT_AUTO` | 自动选择 M split 或 N split |
| `clamp_threads` | `1` | 对太小的任务裁剪无效线程数 |
| `no_reorder_max_m` | `-1` | 使用内部 auto no-reorder 启发式 |
| `n_split_min_b_panel_bytes` | `512 KiB` | B panel 足够大时才倾向 N split |
| `n_split_m12_min_m` | `24` | N split 下 M12 主体启用的 M 下限 |
| `n_split_m12_min_k` | `64` | N split 下 M12 主体启用的 K 下限 |

默认 AUTO 逻辑：

- `T == 1` 时保持 M split。
- `n_tiles < threads` 时保持 M split。
- `M/8 < threads` 时倾向 N split，避免 M 方向不够分。
- `B_panel >= 512 KiB && N >= 2*M` 时倾向 N split，减少多线程重复读 B。
- 小/H2-sized panel 保持 M split 更稳。
- N split 下只在 `M>=24 && K>=64` 时用 M12 主体，否则用 M8/tail，避免极短 K 回退。

可以通过代码配置：

```c
bf16gemm_sve_schedule_t s;
bf16gemm_sve_get_default_schedule(&s);
s.split_policy = BF16GEMM_SVE_SPLIT_AUTO;
s.n_split_min_b_panel_bytes = 512 * 1024;
bf16gemm_sve_set_schedule(&s);
```

也可以临时用环境变量覆盖：

| 环境变量 | 说明 |
| --- | --- |
| `BF16_SVE_SPLIT=m` | 强制 M split |
| `BF16_SVE_SPLIT=n` | 强制 N split |
| `BF16_SVE_SPLIT=old` | 回到旧 split 规则 |
| `BF16_SVE_CLAMP_THREADS=0` | 关闭线程数裁剪 |
| `BF16_SVE_NOREORDER_MAX_M=0` | 强制所有 M 走 A reorder |
| `BF16_SVE_NOREORDER_MAX_M=8` | `M<=8` 跳过 A reorder |

### BF16 NEON 兼容入口

NEON BF16 老路径仍然使用通用头文件：

| 文件 | 作用 |
| --- | --- |
| `bf16gemm.h` | BF16 通用对外接口 |
| `bf16gemm_mt.c` | NEON/通用多线程调度、padding、B pack |
| `bf16gemm_k.S` | NEON BF16 fp32 output kernel |
| `bf16gemm_k_bias.S` | NEON BF16 bias kernel |

NEON 典型链接：

```bash
cc -O3 -fopenmp -mcpu=native your_code.c bf16gemm_mt.c bf16gemm_k.S bf16gemm_k_bias.S -lm
```

### I8 路径

当前目录也包含 I8 GEMM：

| 文件 | 作用 |
| --- | --- |
| `i8gemm.h` | I8 对外头文件 |
| `i8gemm_sve.c` | SVE I8 多线程调度和 wrapper |
| `i8gemm_mt.c` | NEON/通用 I8 多线程调度 |
| `i8gemm_k.S` | NEON I8 主 kernel |
| `i8gemm_k_bias.S` | NEON I8 bias kernel |
| `i8gemm_k_ldc.S`, `i8gemm_k_ldc_lda.S` | I8 ldc/lda 变体 |

## 2. 两套完整测试框架

当前建议把测试分成两套理解：正确性测试框架和性能/归因测试框架。

### 2.1 正确性测试框架

#### SVE 正确性

| 文件 | 作用 |
| --- | --- |
| `test_correctness_sve.c` | SVE BF16/I8 wrapper + dispatch 正确性测试 |
| `bf16gemm_sve.c` | 被测 BF16 SVE 调度 |
| `bf16gemm_sve.S` | 被测 BF16 SVE kernel |
| `i8gemm_sve.c` | 被测 I8 SVE 实现 |
| `bf16gemm.h`, `bf16gemm_sve.h`, `i8gemm.h` | 对外声明 |
| `gemm_params.h` | 参数结构 |

运行：

```bash
make -f Makefile.sve test-sve
```

当前通过标准：

```text
SVE correctness OK: 6000 wrapper+dispatch cases
```

#### NEON/通用正确性

| 文件 | 作用 |
| --- | --- |
| `test_correctness.c` | NEON/通用 BF16/I8 tail-dispatch 正确性测试 |
| `bf16gemm_k.S`, `bf16gemm_k_bias.S` | BF16 NEON kernel |
| `bf16gemm_mt.c` | BF16 NEON/通用调度 |
| `i8gemm_k.S`, `i8gemm_k_bias.S` | I8 NEON kernel |
| `i8gemm_mt.c` | I8 NEON/通用调度 |

如果只关心 SVE 新路径，优先使用 `make -f Makefile.sve test-sve`。

#### M8 接口正确性

| 文件 | 作用 |
| --- | --- |
| `test_sve_m8_interfaces.c` | standalone M8 SVE 接口测试 |
| `bf16gemm_sve_m8_nld.S` | standalone SVE M8 attribution kernel |

这套主要用于验证 M8 standalone benchmark kernel 的接口，不是完整 wrapper 路径。

### 2.2 性能和归因测试框架

#### SVE/NEON M8 parts + full-tail 框架

这是当前最完整的性能归因框架。

| 文件 | 作用 |
| --- | --- |
| `run_m8_parts.sh` | 主脚本：构建、跑 parts/stores/baselines/tail、写 CSV/XLSX |
| `bench_m8_parts.c` | standalone M8 parts 性能测试入口 |
| `bench_full_tail.c` | 完整 dispatch tail 性能测试入口 |
| `bf16gemm_sve_m8_nld.S` | SVE M8 standalone kernel，支持 full/bf16/bias 和 parts 变体生成 |
| `bf16gemm_neon_m8_nld.S` | NEON M8 standalone 对照 kernel |
| `bf16gemm_sve.c`, `bf16gemm_sve.S` | SVE full-tail 被测路径 |
| `bf16gemm_mt.c`, `bf16gemm_k.S`, `bf16gemm_k_bias.S` | NEON full-tail 被测路径 |

常用命令：

```bash
# 跑默认 shape，线程 1/2/4/8，只跑 tail/full dispatch
RUN_PARTS=0 RUN_STORES=0 RUN_BASELINES=0 RUN_TAILS=1 \
THREADS="1 2 4 8" REPS=5 WARMUP=2 RUNS=3 \
KEEP_CSV=1 OUT=/tmp/m8_parts.csv TAIL_OUT=/tmp/m8_tail.csv \
./run_m8_parts.sh

# 跑 standalone M8 parts 归因
RUN_TAILS=0 RUN_BASELINES=1 RUN_PARTS=1 RUN_STORES=1 \
PART_VARIANTS="full nozero nostore noload nozero_nostore noload_nostore" \
STORE_IMPLS="sve neon" STORE_MODES="f32 bf16 bias" \
THREADS="1 2 4 8" \
./run_m8_parts.sh
```

绑核示例：

```bash
THREADS=auto c=0-79 ./run_m8_parts.sh
```

脚本会根据 `c`/`CORES`/`NCORES` 计算可用核心数，并在默认 `SKIP_OVERSUB=1` 时跳过超过可用核心数的线程配置。

默认还会对大 shape 的小线程配置做剪枝：

```text
ABC bytes = A_bf16 + B_bf16 + C_fp32
          = 2*M*K + 2*K*N + 4*M*N
```

当 `ABC > 3MiB` 时，只保留满足 `ceil(ABC / threads) <= 4MiB` 的线程数。小于等于 `3MiB` 的 case 保留全部线程配置。这样跑 80 核 sweep 时，超大矩阵不会继续浪费时间测 1/2/4 核这类每线程数据量过大的点。

相关环境变量：

| 变量 | 默认值 | 说明 |
| --- | ---: | --- |
| `PRUNE_BIG_CASE_THREADS` | `1` | 是否启用大 shape 小线程剪枝 |
| `PRUNE_TOTAL_BYTES` | `3*1024*1024` | 启用剪枝的 ABC 总数据量阈值 |
| `PRUNE_PER_THREAD_BYTES` | `4*1024*1024` | 每线程理论数据量上限 |

关闭剪枝：

```bash
PRUNE_BIG_CASE_THREADS=0 THREADS=auto c=0-79 ./run_m8_parts.sh
```

当前默认 33 shape、`THREADS=auto c=0-79` 时，planned benchmark calls 从未剪枝的 8349 次降到约 6516 次；输出 CSV/XLSX 数据行从 7623 行降到约 5952 行。

#### 常规峰值和 dispatch benchmark

| 文件 | 作用 |
| --- | --- |
| `bench_perf_sve.c` | SVE peak 和 BF16/I8 dispatch benchmark |
| `bench_perf.c` | NEON/通用 peak 和 dispatch benchmark |
| `bench_nld_sve.c` | SVE nld micro benchmark |
| `bf16gemm_sve_nld.S` | SVE nld micro benchmark kernel |

常用命令：

```bash
make -f Makefile.sve bench-sve
make -f Makefile.sve peak-sve
make -f Makefile.sve mt-sve M=64 K=512 N=4096
make -f Makefile.sve bench-nld-sve
```

### 测试结果和说明文档

| 文件 | 说明 |
| --- | --- |
| `bf16_m8_kernel_benchmark_report.md` | M8 kernel 瓶颈和多线程线性度报告 |
| `bf16_m8_dispatch_optimization_result.md` | 本轮 SVE dispatch 调度优化结果 |

CSV/XLSX/log/binary 通常是生成物，不建议提交到 git，除非需要固定某次实验结果。

### 生成物清理建议

常见生成物：

```text
bench_perf
bench_perf_sve
test_correctness
test_correctness_sve
m8_fulltail_*.csv
m8_fulltail_*.xlsx
*.log
*.bak_*
```

这些文件不属于源码路径。提交前建议用 `git status --short` 检查，只提交源码、头文件、脚本和需要保留的文档。
