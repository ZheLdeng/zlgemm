# i8gemm — ACL/KleidiAI gap analysis + optimization plan (2026-06-25, Neoverse-V1 8c)

Machine: AWS Neoverse-V1, 8 cores, SVE VL=256b, i8mm+bf16. (NOT the Huawei 80c box;
Huawei-specific t79 scheduling can't be measured here.)

## 1. Measured gaps vs ACL (i8 -> i32), runs=4 best-of
Data: `results/m8/gap_sweep_sve_vs_acl.csv` (ratio = sve/acl; <1 means ACL wins).
KleidiAI's i8 path emits f32-dequant (different op) so ACL is the real i8->i32 baseline.

SVE already wins decisively on:
- large shapes (>=512 in 2 dims): 1.05–1.15x
- small-M deep-K skew (8/16/32 x 2048+ x *): 1.2–1.8x  (our strong suit)

ACL wins (clusters, worst first):
1. **small-M shallow/medium-K, multithread — `8x512x512`**: 0.57 @t8, 0.68 @t4, 0.73 @t2.
   Does not scale (442->647 over t1..t8). WORST gap. Root cause found (below).
2. **small single-thread cubes**: `128x128x128 t1`=0.71, `32x256x256 t1`=0.83, `16x128x128 t1`=0.86.
3. **K=256,N=256 medium cubes, all threads**: `256x256x256` 0.86–0.93, `64/128x256x256` 0.88–0.94.
4. **N=512 cubes @t8** (bandwidth): `512x512x512`=0.94, `256x512x512`=0.95, `64x512x512`=0.88.

## 2. Root causes (probed with env overrides)
- Gap #1 (`8x512x512`): (a) hybrid no-pack kernel is gated off for K>256
  (`i8_use_hybrid_for_shape`: `K_r<=256`); (b) the MIN_MACS fork-amortization clamp
  (tuned for Huawei 80c) caps this 2.1M-mac shape to 4 threads on the 8c box.
  Forcing hybrid + no clamp + N-split: **647 -> 1456 @t8 (vs ACL 1131, ratio 1.29)**, scales cleanly.
- Gap #3 (`64/128x256x256`): hybrid recovers these (now beat/match ACL). `256x256x256`
  stays in packed path — genuine medium-cube blocking/bandwidth gap (ACL `sve_hybrid_..6x4VL`).
- Gap #2/#4: packed-path single-thread + medium-cube @t8 microkernel/bandwidth efficiency.

## 3. The three optimization opportunities (ranked by GOPS upside)
1. **Small-M multithread scaling** (gap #1, also #3 low-M): biggest, clearest win.
2. **Medium cube @ high threads** (gap #4): N=512 cubes @t8 bandwidth/blocking.
3. **Small single-thread shapes** (gap #2): packed-path/routing efficiency for tiny cubes.

## 4. Part 3 — new M-split lib design (`i8gemm_msplit`)
Goal: a lib parallel to the M+N/2D-grid dispatch that **only partitions the output by M**
(never subdivides N columns across threads), and for **small M** keeps every thread busy by
**splitting the K reduction** over the efficient 8-row tile (the 1/2/4-row kernels waste
compute — they all run the full 8-row smmla machinery — so finer M granularity is a loss;
split-K preserves 8-row efficiency).

Scheme: arrange P threads as (m_groups x k_slices).
- m_groups = min(P, mblk) where mblk = ceil(M/8); each group = a contiguous 8-row-aligned M-band.
- k_slices = ceil(P / m_groups) (>1 only when M is too small to fill P by M alone).
- Each thread computes (its M-band) x (full N) x (its K-slice) into a private partial when
  k_slices>1, then a parallel reduction sums the k_slices partials into C. k_slices==1 writes
  C directly (pure M-split, zero overhead).
Reuses `i8_pack_B`; one small asm kernel (`i8gemm_msk`, hybrid + explicit B n-panel stride so a
K-slice can be swept over full N in a single call). Bit-exact (i32 add associative).

## 5. Round-1 results (implemented + committed, branch i8-msplit-and-smallm-opt)
- **gap #1 fixed** (small-M hybrid routing M<=16,K<=1024 + path-aware fork clamp):
  `8x512x512 @t8` 0.57 -> 1.21x ACL, @t4 0.68 -> 0.94, scales cleanly. No regression
  on M>=24, large shapes, or small-M deep-K (still 1.1-1.8x). 6000 correctness cases pass.
- **msplit lib**: M-split-only + split-K, 7168/7168 bit-exact. Within the M-split-only
  paradigm split-K gives 1.5-2.6x over naive M-split for small M at high threads.
  vs the M+N main lib it is lower for tiny memory-bound shapes (N-split streams each B
  panel once; split-K pays reduction + private-buffer traffic) and matches/wins where
  M-banding fills the threads (64x512 t8 1.03, 128x1024 t8 1.06).

## 6. Stable residual gaps (runs=8, after round-1) and root cause
```
 single-thread:  128x128x128 t1 0.74  32x256x256 t1 0.82  16x128x128 t1 0.84  256^3 t1 0.95
 K=256/N=256 t4: 64x256x256 0.86  256^3 0.88  128x256x256 0.89  32x256x256 0.92  (t8 ~parity, grid mode)
 N=512 cubes t8: 64x512x512 0.87  128x512x512 0.92  256x512x512 0.95  512^3 0.98
```
All residual gaps are **overhead/issue-bound at small-medium sizes**, and trace to one
root cause: our i8 microkernel is **8x2VL (8 rows x 16 cols)** while ACL's winning kernel
is **sve_hybrid_s8s32_mmla_6x4VL (6 rows x 32 cols)**. The wider N-tile halves the number
of N-tile iterations (and per-tile pipeline fill/drain + A-broadcast setup), which is what
ACL amortises better when K/N are small. Scheduling is now essentially optimal here (the
2D-grid mode picks well; N-split/M-split/grid were measured and grid wins for K=256 cubes).

### Recommendation for the deep round-2 lever (needs a decision)
The only remaining lever is a **wider (4VL) i8 microkernel**. Caveats before investing:
- It is a large, correctness-risky asm effort (accumulator pressure: 6x4VL uses most of the
  z-register file).
- It is **V1-specific**. On the Huawei target smmla issues at **1/cycle (half of V1)**, so
  medium/small cubes there are compute-bound at the smmla unit, not issue/overhead-bound --
  a V1-tuned wider kernel may give little or nothing on Huawei. The round-1 scheduling wins
  (gap #1, msplit split-K) are hardware-portable; the microkernel-width gap is not.
Suggest validating the bottleneck on Huawei (perf: smmla issue rate vs stall_backend on a
128^3 single-thread run) before building a 4VL kernel.

---

# 附录 A — i8 SVE 优化历程（step-by-step，整合自已删除的旧报告）

> 本附录把 2026-06-08…06-24 多份中间报告里**有对比性的 before/after 数据**整合到一处，
> 以体现逐步优化效果，原始中间报告与中间 probe CSV 已删除（数据可由本表追溯）。
> 平台：Neoverse-V1 8c，单核峰值 i8 661 GOPS / bf16 331 GFLOPS（2-op 口径）。

## A.1 i8 SVE 优化时间线（每步 before→after，均 bit-exact、无回退）

| # | 优化 | 代表形状 | before | after | 收益 |
|---|---|---|---|---|---|
| 1 | **修正 96³ 崩溃 + m12 i32 数值 bug**（`BUILD_I32_OFFSET_BASE` scratch z0→z2）| M≥12,K≥64,n_tiles>1 | 崩溃/结果错 | 正确 | 正确性（之前静默错误）|
| 2 | **存储反交织**（scatter st1w 实测比连续慢 7.2×；trn1/trn2+ext）| 256×16×256 t1 | 64 | 131 | +106% |
| | | 128³ t1 | 258 | 333 | +29% |
| | | 256³ t1 | 373 | 454 | +22% |
| | | 512³ t1 | 482 | 526 | +9% |
| | | 2048×4096×8192 t8 | 2727 | 2842 | +4% |
| 3 | **线程本地缓存 A-pool**（去掉每次调用 malloc/free）| 小形状 | — | — | 去 per-call 开销 |
| 4 | **hybrid 免 pack 微内核**（小形状直读 A）| 64³ t1 | 139 | 276 | +99% |
| | | 64³ t8 | 92 | 360 | +289% |
| | | 64×128×64 t8 | 172 | 602 | +250% |
| 5 | **线程感知 hybrid 路由**（多线程中等形状）| 96³ t8 | 386 | 771 | 超 ACL 669 |
| | | 128³ t8 | 525 | 1236 | 超 ACL 956 |
| 6 | **大 N 偏斜放开 N 限制**（nt≥2）| 64×64×512 t8 | 446 | 1110 | 超 ACL 1058 |
| 7 | **中等立方体 2D 自适应分发 @8**（vs ACL 比值）| 160³/192³/200³/224³/256³ | .86/.94/.80/.82/.84 | .99/.98/.95/.94/.89 | — |
| 8 | **(06-25) 小 M hybrid 路由 + 路径相关 clamp** | 8×512×512 t8 | 0.57×ACL(624) | 1.21×ACL(1344) | 修复最差点 |
| | | 8×512×512 t4 | 0.68× | 0.94× | |
| 9 | **(06-25) msplit lib（split-K）** | 8×2048×2048 t8 | 545(朴素M-split) | 1430(split-K) | 2.62× |

## A.2 单核利用率天花板（硬件事实，durable）
| 场景 | i8 | bf16 |
|---|---|---|
| 纯 compute（操作数常驻、无 load）| 663 GOPS = **100%** | 332 = 100% |
| 预排 + L1 常驻、仅 kernel、m12 tile | 554 = **83.8%** | 286 = 86.5% |
| 16 smmla + 8 loads（intensity 2.0）微环 | **92.3%** | — |
| 完整 dispatch 单线程大形状 | 554 = 83.8% | 278 = 84% |

**结论**：32 个 z 寄存器下最大可用 tile 6×4（24 累加器）→ 2.4 mmla/load，预排 kernel 上限约 **87%**
（已基本吃满），**V1 上 95% 不可达**；compute 不是杠杆。NEON 预排上限（reo_ld 8×8）实测仅 78.4%
< SVE m12 89.7%，故"SVE 应改走 NEON"假设被推翻（V1 上 SVE 微内核更优）。

## A.3 三方单核对比 digest（480 形状，threads=1，three_lib_grid_full_reps1）
中位（GOPS）：i8 — SVE 318.8 / NEON 266.5 / **ACL 429.0** / KleidiAI 339.4；
bf16 — SVE 204.8 / **ACL 220.2** / KleidiAI 203.0。
按 cache 分层：ACL 在 L2/H2 中小规模强（kernel selector + hybrid 6x4VL）；SVE 在 GT_L2 大规模
追平/超 ACL（i8 GT_L2: SVE 416.6 ≈ ACL 466.2；bf16 GT_L2: SVE 235.6 > ACL 232.1）；KleidiAI 在
L1 短路径强但输出是 f32 dequant（非同口径）。**bf16 多线程 i8gemm 在所有点领先 ACL 和 KleidiAI。**

## A.4 尾块（tail）durable 结论
尾块 +4 ≈ 填满半个 M8 block，普遍 ≥99%；痛点是 task 粒度不均：**SVE 痛在低线程**（512×4096×1024
@t8 +1 = 79.8%），**NEON 痛在中线程**（@t16/t32 ~78–92%）；≥10 线程或 +4 偏移基本无损。
建议：低线程 + 小 tail（+1/+2）把尾块并入最后一个 M8 block（padding）而非单独平铺切分。

## A.5 80 核 compute-only 线性度（华为机器，详见 huawei_80c 交接）
SVE i8 `compute_only` 到 64 核近线性（~11451 GOPS，效率~1.0），80 核回落到 0.58；NEON i8 在
64/80 核扩展崩塌（0.47/0.19）。→ 硬件到 64 核近线性，掉速是 kernel 调度问题（已在 huawei 交接处理）。

## A.6 其它 durable 对比（整合自 80c lite / kleidiai 实验，原报告已删）
- **80c split 敏感性（NEON）**：`512×4096×1024` 一类在高线程下 full/compute 天花板偏低（8T~54%），
  改 **m-split** 避免在线 A 重复 pack 后能逼近天花板，实测最大提升 **2.7×@32T**（NEON I8）、
  1.7–2.7×（NEON BF16）。→ auto 已加入"大 M/K 偏向 m-split"规则。SVE I8 full 利用率稳定 45–52%
  （compute-only 峰值高，更早进入 load/store 受限）。
- **NEON m16n4 实验路径**（`i8gemm_mt_dispatch_m16n4`，packed-A 复用 + K32 双缓冲）：GT_L2 中位
  逐版提升 `1c 170→245.7, 2c 311→402.7, 4c 537→666.2, 8c 790→885.2`，但仍 < KleidiAI（手写 asm +
  N4 专用 pack）且 < 默认 8×8（宽 N 大矩阵）。结论：m16n4 仅适合**窄 N / 小 M tail / 多线程 work 不足**
  场景，宽 N 大矩阵继续用 8×8；未纳入默认 selector。
- **NEON compute-only 异常**：≥4 核 NEON I8 compute-only 仅 ~67%/core（参考应 ~100%），疑似绑定/频率
  策略；若修复 full 路径绝对性能会同步提升（待查，低优先级）。


---

# 附录 B — 华为 80c 细粒度利用率扫描结论 (2026-06-25, fine_sweep, n=247)

数据：`results/m8/fine_sweep.log`（脚本 `tests/huawei_fine_sweep.sh`）。peak=370.9 GOPS/核(2-op)，e2e(含 A-pack)。

- 单核利用率 4.6–96.2%（均值 65.8）；最优线程利用率 0.2–90.3%（均值 30.6）。
- **单核由 K 主导**：K16~20% / K64~50% / K256~78% / K≥1024 90–96%；其次小 M（M1~11% M2~22% M4~44% M8~88%）；再 N（N8~26% N16~52% N17~39%(pad32) N≥256 90%+）。大 K + M≥8 + N≥256 已贴微内核天花板。
- **大形状多核 75–90%**（4096³=90.3%），带宽墙比 V1 高，不是主要问题。
- **最大免费杠杆=线程过度并行**：GOPS-max 几乎总 t64，但拐点线程小得多、GOPS 几乎不变、利用率高 2–4×。例：32×4096×1024 t64=4507(19%) vs t16=4491(75.7%)；8×4096×1024 t48=2964(21.6%) vs t16=3734(62.9%)。最优线程从不取 79/80。

四项优化（用户批准全做）：
1. 线程拐点选择（effective_threads 收敛中等形状到拐点；免费提利用率 2–4×、释放核）。
2. 小 K / 中 K 路由到 hybrid 免 pack（K128~64% K256~78% 的 A-pack 开销）。
3. 窄 N / 非 16 对齐（N=17 pad 到 32 的悬崖）。
4. decode M=1–4 GEMV 路径（8 行 smmla tile 结构浪费，需 sdot/dot kernel）。


---

# 附录 C — 四项优化最终结论（数据驱动，2026-06-25）

华为实测（bw_probe / gemv_clamp）后，关键结论：**微内核在好区间已 84–96%，剩余弱点全是调度，无需写新微内核。**

| # | 方案 | 结论 | 落地 |
|---|---|---|---|
| 1 | 线程拐点 | ✅ `i8gemm_recommend_threads` advisory（opt-in，`I8_SVE_AUTO_THREADS=1`），floor `I8_REC_MACS` 按 fine_sweep knee 校准 | commit 85588ae |
| 2 | 小K→hybrid | ❌ 实测回退（packed 对 K≥64 全 M 更优）；小K低util是结构性短K流水线，走 #1 | 不做（数据） |
| 3 | 窄N内核 | ❌ 单核已~84%，非内核问题；多线程低util=M-split 重复 streaming B（cache-blocking），同中cube | 不做内核；属调度 |
| 4 | GEMV内核 | ❌ 不写内核（M=1 内存受限，DRAM≥766GB/s，smmla 浪费被内存墙掩盖）；✅ 改为**路由 M≤4 到 hybrid**（华为 1.2–1.6×） | commit a71e36d |

华为 gemv_clamp 实测：1×4096×1024 default 266→hybrid 437；1×4096×4096 780→1164；1×11008×4096 1245→1610；1×4096×11008 1300→1541。

## #1 模型校准（拟合 247 形状 fine_sweep knee）
`recommend_threads` 改为 **B-panel 门控**模型（commit 见下）：`t=min(maxT, work_units)`，仅当
B 面板 `K_r*N_r > I8_REC_CACHE`(默认 2Mi) 时再按 `macs/I8_REC_MACS`(默认 512Ki) 收缩。
对 247 真实 knee 拟合：**~97% 形状 pred≥knee（不丢 GOPS）**、far-below<knee/2 仅 2–3%、平均
用 ~54 核（vs 64）。纯 macs-floor（无 B 门控）只有 76% 安全、19% 丢 GOPS，故采用 B 门控。
局限：仍**抓不住带宽饱和的中等形状**（如 32×4096×1024 knee=16，macs 够大过不了 floor）——
其 knee 由 B-streaming 速率决定、非总 macs；精确 knee 需 per-machine sweep 调 `I8_REC_MACS/CACHE`。

**真正剩余的硬骨头（均为调度/带宽，非微内核）**：
- (a) 中等形状过度并行：#1 advisory 已给，默认全自动需华为带宽/LLC 模型（单一 floor 区分不了"带宽饱和中等形状"vs"可在N上扩展的小B形状"）。
- (b) 窄N + 大M + 大K 的 B 重复 streaming：需 ACL 式 2D/K cache-blocking（大工程）。
- 微内核本身（compute）已非瓶颈。
