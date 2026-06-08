# GEMM Multi-threading 性能对比分析

## 测试环境
- SVE VL=256 bits, segments=2, N tile=16
- max OMP threads=8
- 编译器: GCC with -O3 -fopenmp -mcpu=native

## 一、OpenMP 调度方式分析

### 当前调度: `schedule(static)` — 已是最优选择

对于 GEMM 而言，每个 tile 的计算量完全一致（相同的 M×K×N 乘积），工作负载天生均匀：

| 调度方式 | 适用场景 | GEMM 适用性 |
|---------|---------|------------|
| `schedule(static)` | 均匀负载 | ✅ **最佳** — 零运行时开销，完美负载均衡 |
| `schedule(dynamic, N)` | 不均匀负载 | ❌ 每次取 chunk 需要原子操作，增加开销 |
| `schedule(guided, N)` | 递减负载 | ❌ 同 dynamic，chunk 大小递减增加复杂度 |
| `schedule(auto)` | 编译器决定 | ✅ 通常映射为 static，无额外收益 |

**结论**: 换调度方式不会改善性能，问题根源不在调度策略，而在**线程团队的创建/销毁开销**。

### 真正的问题：重复的 parallel region

SVE 路径 (`bf16gemm_sve.c`) 在一个调用中创建了 **多个 parallel region**：
- `bf16_prepare_A_reorder_pool()` → 第一个 `#pragma omp parallel for`
- M12 block reorder → 第二个（如果有 M12 blocks）
- M12 block dispatch → 第三个
- Tail block reorder → 第四个
- Tail block dispatch → 第五个

每次 parallel region 都意味着线程唤醒 → 执行 → barrier → 休眠的完整周期。

## 二、性能数据

### 线程管理开销（空 dispatch）

| 线程数 | OpenMP (us) | Thread Pool (us) |
|-------|-------------|-----------------|
| 1 | ~0 | 8.6 |
| 2 | ~0 | 11.9 |
| 4 | ~0 | 15.4 |
| 8 | ~0 | 28.4 |

> OpenMP 的 ~0us 是因为 libgomp 内部已有线程池，线程处于 spin/park 状态，唤醒极快。
> Thread pool 的 8.6-28.4us 是 mutex+condvar 的固定开销。

### Large Shape (M=512, K=1024, N=4096)

| dtype | threads | variant | best_sec | GFLOPS | throughput |
|-------|---------|---------|----------|--------|------------|
| BF16 | 1 | original | 0.01516 | 283 | 280 |
| BF16 | 1 | fused_omp | 0.01578 | 272 | 270 |
| BF16 | 1 | threadpool | 0.01582 | 272 | 274 |
| BF16 | 4 | original | 0.00524 | **820** | 798 |
| BF16 | 4 | fused_omp | 0.00602 | 713 | 637 |
| BF16 | 4 | threadpool | 0.00647 | 664 | 636 |
| BF16 | 8 | original | 0.00282 | **1525** | 1446 |
| BF16 | 8 | fused_omp | 0.00351 | 1223 | 1130 |
| BF16 | 8 | threadpool | 0.00352 | 1220 | 896 |
| I8   | 8 | original | 0.00223 | **1922** | 1748 |
| I8   | 8 | fused_omp | 0.00225 | **1910** | 1781 |
| I8   | 8 | threadpool | 0.00252 | 1704 | 1472 |

### Medium Shape (M=128, K=512, N=1024)

| dtype | threads | variant | GFLOPS | throughput |
|-------|---------|---------|--------|------------|
| BF16 | 8 | original | **1545** | 1231 |
| BF16 | 8 | fused_omp | 1470 | 481 |
| BF16 | 8 | threadpool | 1111 | 714 |
| I8   | 8 | original | **1837** | 1496 |
| I8   | 8 | fused_omp | 1766 | 1479 |
| I8   | 8 | threadpool | 1130 | 698 |

### Small Shape (M=64, K=256, N=512)

| dtype | threads | variant | GFLOPS | throughput |
|-------|---------|---------|--------|------------|
| BF16 | 8 | original | 701 | 542 |
| BF16 | 8 | fused_omp | **1146** | **1010** |
| BF16 | 8 | threadpool | 419 | 349 |
| I8   | 8 | original | **1076** | 895 |
| I8   | 8 | fused_omp | **1094** | 895 |
| I8   | 8 | threadpool | 404 | 318 |

### Tiny Shape (M=16, K=64, N=128)

| dtype | threads | variant | GFLOPS | throughput |
|-------|---------|---------|--------|------------|
| BF16 | 8 | original | 46 | 45 |
| BF16 | 8 | fused_omp | **105** | **77** |
| BF16 | 8 | threadpool | 10 | 9 |
| I8   | 8 | original | **141** | **128** |
| I8   | 8 | fused_omp | **143** | **128** |
| I8   | 8 | threadpool | 9 | 9 |

## 三、关键发现

### 1. Fused OpenMP 的效果
- **Tiny shapes**: BF16 性能翻倍（46→105 GFLOPS），因为消除了双重 parallel region 开销
- **I8 所有 shapes**: 与 original 持平（I8 路径没有 M12 kernel，融合无负面影响）
- **BF16 Large shapes**: 比 original 慢 20%（缺少 M12 kernel 的使用）
- **实际收益**: 对小模型推理（batch size 小）有明显帮助

### 2. Thread Pool 为什么比 OpenMP 慢？

三个主要原因：

a) **单线程 A-reorder**：当前 thread pool 实现中，A-reorder 是串行的（主线程完成），而 OpenMP 是并行化的。对于 Large shape（M=512, K=1024），A-reorder 本身就需要十几毫秒。

b) **Mutex+Condvar 开销**：每次 dispatch 至少 28μs（8 线程），对小 shape 是灾难性的（tiny shape 总计算只需 ~4μs）。

c) **libgomp 已经很强了**：GCC 的 libgomp 内部维护了高效的线程池，线程创建后通过 spin-wait + adaptive parking 实现纳秒级的唤醒。手写的 pthread 线程池很难超越。

### 3. OpenMP 的最佳实践建议

```
环境变量调优:
  OMP_WAIT_POLICY=active   # 线程 spin-wait 而非 sleep（降低唤醒延迟）
  OMP_PROC_BIND=close      # 线程绑定核心（减少迁移开销）
  OMP_DYNAMIC=false        # 禁用动态线程数调整

代码调优:
  1. 使用 #pragma omp parallel（不用 parallel for）+ 手动工作分配（NEON path 模式）
  2. 将 A-reorder 融入同一个 parallel region
  3. 对外层多次调用场景，考虑单一 persistent parallel region
```

## 四、文件清单

| 文件 | 作用 |
|------|------|
| [gemm_thread_pool.h](gemm_thread_pool.h) | Thread pool API 头文件 |
| [gemm_thread_pool.c](gemm_thread_pool.c) | Thread pool 实现（pthread + condvar） |
| [bench_thread_pool.c](bench_thread_pool.c) | 三路对比 benchmark |
| [Makefile.sve](Makefile.sve) | 新增 `make bench-tp` 目标 |

运行 benchmark:
```bash
make -f Makefile.sve bench-tp
```
