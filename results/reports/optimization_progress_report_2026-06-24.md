# i8gemm/bf16gemm 优化进度测试报告（2026-06-24）

> 数据来源：`tests/run_three_lib_grid.py`（三方性能对比网格）+ `tests/run_m8_parts.sh`
> （i8 kernel 耗时构成 attribution）。平台 Neoverse-V1，8 核 @2.6GHz，SVE(非SVE2)
> VL=256bit，i8mm+bf16。i8 单核峰值 **661 GOPS**，bf16 单核峰值 **331 GFLOPS**。
> 原始 CSV：`results/m8/three_lib_grid_report_2026-06-24.csv`、
> `results/m8/i8_parts_report_2026-06-24.csv`（及同名 .xlsx）。

---

## 0. 一句话结论

- **当前 SVE kernel 已基本追平 ACL**：i8 单线程 geomean **0.95**（中位 0.98），8 线程
  geomean 0.87 / 中位 **0.96**；bf16 单线程 geomean 0.93 / 中位 **1.00**，8 线程中位 0.95。
  对 KleidiAI：i8 全胜（geomean 2.2×），bf16 geomean 1.2×。
- **耗时构成**显示：单线程效率瓶颈 = **小 K 时的 A-reorder/流水线填充开销**（K=128 占 42%，
  K≥1024 降到 ≤10%）；**store 几乎免费**（反交织连续写 ≈0%），load 占比小（缓存内 0–4%）；
  compute-only 上限 ~**91%**（与 priority #1 实测的 89.7% 预排上限一致）。
- **8 线程绝对利用率只有 ~55–60%**（峰值 5288 GOPS），且 ACL 同样如此——这是
  **多核 LLC/DRAM 带宽墙**，不是微内核问题。

---

## 1. 测试方法

| 项 | 设置 |
|---|---|
| 三方网格 | `run_three_lib_grid.py`，M∈{16,64,256,1024} K∈{128,512,2048} N∈{64,256,1024,4096}，线程∈{1,8}，RUNS=3 best-of，绑核 0-7 |
| 对比对象 | `i8gemm_sve`（本仓库当前态，含本轮 2D 分发优化）、`acl_auto`(ACL)、`kleidiai_*` |
| 耗时构成 | `run_m8_parts.sh RUN_I8_PARTS=1 I8_PART_IMPLS=sve`，变体 full/nostore/noload/compute_only，单线程，RUNS=4 best-of |
| 构成口径 | 各变体在 i8gemm_sve.S 上 ablation：`full`=完整内核；`nostore`=去 C 写；`noload`=去 A/B 读（置1）；`compute_only`=仅 smmla。时间份额由 1/GFLOPS 反推 |

> 注：耗时构成测的是 SVE `i8gemm_k_ld`（在途 A-reorder 的 load-path 内核，生产 tail 块在用）。
> 生产主路径的 `i8gemm_k_nld_m12` 把 reorder 提前到 pack 阶段，故其 "other%" 会更低。
> NEON 侧 attribution 因 `bench_i8_parts.c` 链接缺符号（`i8gemm_k_reo_ld`）未跑，与本轮无关。

---

## 2. 当前 kernel 性能（三方网格）

### 2.1 i8gemm_sve vs ACL / KleidiAI（96 个形状点/数据类型）

| dtype | 对比 | 线程 | geomean | 中位 | min | max | 追平率(≥0.98) |
|---|---|---|---|---|---|---|---|
| i8 | vs ACL | 1 | 0.952 | 0.981 | 0.567 | 1.304 | 25/48 |
| i8 | vs ACL | 8 | 0.868 | 0.957 | 0.286 | 2.065 | 20/48 |
| i8 | vs KleidiAI | 1+8 | **2.185** | 1.821 | 1.246 | 9.751 | 96/96 |
| bf16 | vs ACL | 1 | 0.929 | **1.001** | 0.515 | 1.260 | — |
| bf16 | vs ACL | 8 | 0.899 | 0.949 | 0.427 | 1.204 | — |
| bf16 | vs KleidiAI | 1 | 1.071 | 1.077 | 0.843 | 1.433 | — |
| bf16 | vs KleidiAI | 8 | **1.397** | 1.308 | 0.584 | 3.880 | — |

**解读**：
- **单线程**几乎与 ACL 平手（i8 中位 0.98、bf16 中位 1.00），离微内核上限很近。
- **8 线程中位仍 0.96**，但 geomean 被一批**小形状**离群点拉低（见 2.3）。
- 对 KleidiAI：i8 全面领先，bf16 多数领先。

### 2.2 绝对峰值利用率（i8gemm_sve, i8）

- 单线程最高：M256 K2048 N4096 = **579 GOPS（88% of 661）**；K2048 一档普遍 86–88%。
- 8 线程最高：M1024 K2048 N4096 = **3176 GOPS（60% of 5288）**；最好的几个都在 55–60%。
- → 单核接近 roofline；**8 核只有 ~60%**，且 ACL 亦然（比值 ~0.9），即**带宽墙**。

### 2.3 强项 / 弱项（vs ACL）

弱项（全部是 **8 线程下的小形状**，绝对量很小、线程切分开销占主导）：

| 形状 | ours | ACL | 比值 |
|---|---|---|---|
| M16 K128 N4096 t8 | 496 | 1735 | 0.29 |
| M64 K512 N256 t8 | 899 | 2114 | 0.43 |
| M16 K128 N1024 t8 | 473 | 1048 | 0.45 |
| M16 K2048 N64 t8 | 395 | 791 | 0.50 |

强项（小形状/大偏斜，本仓库反超）：

| 形状 | ours | ACL | 比值 |
|---|---|---|---|
| M16 K128 N64 t8 | 309 | 150 | 2.07 |
| M16 K2048 N4096 t8 | 2620 | 1697 | 1.54 |
| M64 K128 N64 t8 | 535 | 397 | 1.35 |
| M256 K2048 N1024 t8 | 3041 | 2498 | 1.22 |

### 2.4 中等立方体 @8（本轮 2D 分发优化的专项，见 `i8_medium_cube_2d_blocking_2026-06-24.csv`）

| S³ @8 | 优化前/ACL | 优化后/ACL |
|---|---|---|
| 160³ | 0.86 | **0.99** |
| 192³ | 0.94 | **0.98** |
| 200³ | 0.80 | **0.95** |
| 224³ | 0.82 | **0.94** |
| 256³ | 0.84 | **0.89** |

---

## 3. Kernel 耗时构成（SVE i8，单线程，% 为 FULL 内核的时间份额）

| 形状 | cache | full(GOPS) | %峰值 | compute_only | %峰值 | store% | load% | other%* |
|---|---|---|---|---|---|---|---|---|
| M64 K128 N256 | H2 | 276 | 41.8 | 506 | 76.5 | 0.0 | 3.7 | **41.7** |
| M64 K256 N256 | H2 | 363 | 55.0 | 535 | 81.0 | 0.5 | 1.7 | 29.9 |
| M64 K512 N256 | H2 | 454 | 68.7 | 568 | 85.9 | 0.0 | 2.1 | 17.9 |
| M64 K1024 N256 | H2 | 520 | 78.7 | 587 | 88.8 | 0.1 | 1.1 | 10.2 |
| M64 K2048 N256 | L2 | 556 | 84.2 | 598 | 90.4 | 0.0 | 1.4 | 5.5 |
| M64 K4096 N256 | GT_L2 | 528 | 79.9 | 601 | 91.0 | 2.6 | 8.8 | 0.9 |
| M64 K512 N64 | L1 | 472 | 71.3 | 578 | 87.5 | 0.0 | 0.0 | 18.5 |
| M64 K512 N1024 | L2 | 432 | 65.3 | 566 | 85.6 | 1.3 | 6.2 | 16.1 |
| M256 K256 N256 | H2 | 364 | 55.0 | 533 | 80.6 | 0.0 | 1.5 | 30.2 |
| M256 K2048 N512 | GT_L2 | 516 | 78.0 | 595 | 90.0 | 3.6 | 7.8 | 2.0 |

\* other% = A-reorder + zero/accumulate + 循环/流水线填充（= 100 − store − load − compute）。

**关键发现**：
1. **compute_only 上限 ~91%**（K≥2048）：smmla 流水线本身的天花板，和 priority #1 实测的预排
   m12 上限 89.7% 吻合——微内核已接近极限，不应再在 compute 上抠。
2. **other%（reorder+填充）是单线程的主瓶颈，且强烈依赖 K**：K128 占 **42%** → K256 30% →
   K512 18% → K1024 10% → K2048 5.5%。即**短 K 摊不开 A-reorder 与流水线填充**。
   → 生产主路径用 `i8gemm_k_nld_m12`（reorder 提前到 pack）正是针对此；小 K 形状走 hybrid（免 pack）。
3. **store 几乎免费**（缓存内 ≈0%）：印证反交织连续写（trn1/trn2 + ext，替代 scatter st1w）的收益。
   仅在 GT_L2（K4096 / 大 N）C 写落到内存时升到 2.6–3.6%。
4. **load 在缓存内很小（0–4%）**，但在 GT_L2 升到 ~8%——B streaming 触及内存带宽。
5. **最佳工作点在 L2（K2048，84%）**；再大（GT_L2/K4096）反而回落到 80%，因为 B/C 流量
   超出 L2，开始受内存带宽限制（load 8.8%）——与第 2.2 节"8 核带宽墙"同源。

---

## 4. 结论与下一步

**现状**：单线程已接近 roofline（i8 中位 0.98 vs ACL，峰值 88%）；微内核 compute 上限 ~91% 已基本吃满；
store/load 在缓存内开销极低。剩余两块差距都指向**带宽 / 调度**而非 compute：

1. **多核带宽墙**：8 线程绝对利用率 ~60%（ACL 同档），中大立方体经本轮 2D 分发已收窄到 0.89–0.99。
   256³ 仍 0.89，是 A+B 的 LLC 聚合带宽 / prefetch 瓶颈。
   *下一步*：K-blocking 让 A/B panel 更常驻、软件 prefetch B 下一 panel、更宽 N tile（贴近 ACL 4VL）。
2. **小形状 @8 的线程切分开销**：M16/M64 被切 8 份时 ACL 更省（部分点 0.3–0.5）。
   *下一步*：对 ABC 字节数过小的多线程 case 降低并行度（dispatch 里已有 `i8_sve_effective_threads`
   雏形，可按本节弱项数据再调阈值）。
3. **短 K 的 reorder 开销**：已由 m12（reorder 前移）+ hybrid（免 pack）覆盖；无需再动微内核。

> 复现：
> `OUT=results/m8/three_lib_grid_report_2026-06-24.csv RUNS=3 CORESET=0-7 python3 tests/run_three_lib_grid.py`
> （注：本次把 grid 的 M/K/N/threads 常量临时收窄为代表性子集，跑完已还原）；
> `RUN_I8_PARTS=1 I8_PART_IMPLS=sve I8_PART_VARIANTS="full nostore noload compute_only" \`
> `CASES="..." THREADS=1 RUNS=4 KEEP_CSV=1 OUT=results/m8/i8_parts_report_2026-06-24.csv bash tests/run_m8_parts.sh`
> （SVE parts 需用 wrapper 补 `i8gemm_k_reo_ld` 链接桩，详见报告 §1 注）。
