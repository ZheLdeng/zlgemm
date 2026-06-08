# Folder Usage

日期：2026-06-08

当前目录按用途拆成三部分。

## 1. `lib/`

生产库目录，包含可以封成 `.a` / `.so` 的源码。

主要文件：

| 文件 | 说明 |
| --- | --- |
| `bf16gemm.h` | BF16 通用对外 API |
| `bf16gemm_sve.h` | SVE BF16 调度配置和命名 API |
| `i8gemm.h` | I8 通用对外 API |
| `gemm_params.h` | C/ASM 共享参数结构 |
| `bf16gemm_sve.c` | SVE BF16 wrapper、多线程调度、pack/reorder |
| `bf16gemm_sve.S` | SVE BF16 kernel |
| `i8gemm_sve.c` | SVE I8 实现 |
| `bf16gemm_mt.c`, `i8gemm_mt.c` | legacy NEON 多线程调度 |
| `bf16gemm_k*.S`, `i8gemm_k*.S` | legacy NEON kernel |
| `Makefile` | 生成静态库和动态库 |

构建当前 SVE 库：

```bash
make -C lib
```

输出：

```text
lib/build/libi8gemm_sve.a
lib/build/libi8gemm_sve.so
```

构建 legacy NEON 库：

```bash
make -C lib neon
```

输出：

```text
lib/build/libi8gemm_neon.a
lib/build/libi8gemm_neon.so
```

注意：SVE 和 NEON 版本都导出 `bf16gemm_mt` / `i8gemm_mt` 这类同名 API，所以不要把两个库同时链接进同一个程序。

## 2. `tests/`

测试和 benchmark 目录。

主要文件：

| 文件 | 说明 |
| --- | --- |
| `test_correctness_sve.c` | SVE BF16/I8 correctness |
| `test_correctness.c` | legacy NEON correctness |
| `bench_perf_sve.c` | SVE dispatch benchmark |
| `bench_perf.c` | legacy NEON benchmark |
| `bench_m8_parts.c` | standalone M8 parts benchmark |
| `bench_full_tail.c` | full-tail benchmark |
| `run_m8_parts.sh` | M8 parts/store/tail 主脚本 |
| `bf16gemm_sve_m8_nld.S` | SVE M8 standalone test kernel |
| `bf16gemm_neon_m8_nld.S` | NEON M8 standalone test kernel |
| `bf16gemm_sve_nld.S` | SVE compute-only probe |
| `shape.csv` | benchmark shape 输入 |
| `experimental/` | thread pool、旧 nld/nopld、ldc/lda 实验文件 |

构建测试：

```bash
make -C tests
```

运行 correctness：

```bash
make -C tests test-sve
make -C tests test-neon
```

运行 M8 benchmark：

```bash
cd tests
THREADS=auto c=0-79 RESULTS_XLSX=../results/m8/m8_80c_shape_sweep.xlsx ./run_m8_parts.sh
```

`run_m8_parts.sh` 默认会把 `OUT` / `TAIL_OUT` / `RESULTS_XLSX` 写到 `../results/m8/`。

## 3. `results/`

报告和保留的 benchmark 结果。

主要文件：

| 文件 | 说明 |
| --- | --- |
| `reports/bf16_m8_kernel_benchmark_report.md` | M8 kernel report，含 SVE/NEON 对照 |
| `reports/bf16_m8_dispatch_optimization_result.md` | SVE dispatch 优化记录 |
| `reports/FOLDER_USAGE.md` | 本说明 |
| `m8/` | 当前保留的 M8 benchmark CSV/XLSX |

旧的重复结果已经清理，只保留当前有用数据。
