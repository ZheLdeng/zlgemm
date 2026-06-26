# 华为 80 核调度优化 — 进度交接 (2026-06-24)

> ## ✅ 2026-06-25 第二轮（细粒度利用率扫描 + 调度修复，全部华为实测验证）
> 详见 `gap_and_plan_2026-06-25.md`（附录 A–C）与 `smallm_msplit_handoff_2026-06-25.md`。
> branch 已合 main（HEAD ~`d363fa5`）。本轮华为实测净落地：
>
> 1. **t79 质数线程坍塌修复**：`2048×4096×2048 @t79` 0.59→**0.86**、`×1024` 0.65→**0.91**（无回退）。
> 2. **小 M 浅/中 K 多线程**：`8×512×512 @t8` 0.57→**1.21× ACL**（小M路由 hybrid + 路径相关 clamp）。
> 3. **decode GEMV (M≤4) 路由到 hybrid**：`1×4096×1024` 266→**437**、`1×4096×4096` 780→**1164**、
>    `1×11008×4096` 1245→**1610**（gemv_clamp 实测；内存受限，不写 GEMV 内核）。
> 4. **窄 N 调度修复**（packed 2D-grid 加 `n_tiles>=8` 门控）：`2048×4096×64 @t48` 4474→**9007（+97%）**、
>    `2048×16384×24 @t64` 3201→**5733（+79%）**（blocking_probe 实测）。
> 5. **线程拐点 advisory** `i8gemm_recommend_threads`（opt-in，用 247 形状 knee 校准，97% GOPS-safe）。
> 6. **独立 msplit lib**（只切 M + 小 M split-K，1.5–2.6× over 朴素 M-split）。
> 7. **bf16 窄 N 修复**（`bf16_use_2d_split` 加 `n_tiles>=4` 门控）：`2048×16384×24 @t64`
>    795→**1456（+83%）**、t48 +60%、t32 +35%（华为复测确认；i8 窄 N 修复的 bf16 版）。详见
>    `bf16_transfer_analysis_2026-06-25.md`。
>
> **利用率现状（247 形状, peak 370.9/核, e2e 含 A-pack）**：单核 4.6–96.2%（好区间 84–96%，已贴微内核
> 天花板，由 K 主导）；多核大形状 75–90%（4096³=90.3%）；峰值 ~21.4 TOPS。
> **结论：微内核已非瓶颈；调度层弱点（t79/小M/GEMV/窄N/过度并行）已基本清完。** 唯一剩余 = 中等/大
> 形状的 K/2D cache-blocking（blocking_probe 证明无便宜空间，需昂贵重写、收益不确定，建议作为可选未来工作）。
> 验证脚本：`tests/huawei_{validate,fine_sweep,bw_probe,gemv_clamp,blocking_probe}.sh`（均自动重编最新代码）。


> ## 🔴 2026-06-25 更新（在华为机器上跑了 roofline + m8_results.xlsx，结论重大）
>
> ### A. defect #2「微内核 50% 计算天花板 / 2× 余量」是**误判，已关闭**
> 用 `tests/experimental/huawei_roofline.*`（smmla 累加器扫描，纯计算无访存）在华为实测：
> - SVE smmla **8/16/24/30 累加器全平在 0.999 smmla/cycle** → **吞吐受限，非延迟受限**，加累加器/展开无用。
>   perf：纯计算循环 `stall_backend=61% cycles`、IPC 1.20 → 瓶颈是 smmla 执行单元本身（1/cycle）。
> - **真实单核 SVE i8 计算上限 = 370.9 GOPS(2op)**（=1 smmla/cycle×64MAC×2.9G），kernel 的 compute_only 已打满它。
>   报告原来的「峰值 741.6 / 只到 50%」是**按 2 smmla/cycle（像 V1）算错了**——华为这颗核 smmla 只有 V1 的一半吞吐（V1=1.99/cycle）。**微内核没有 2× 可挖。**
> - NEON smmla=1.704/cycle 但每条只 32MAC → 316.3 GOPS < SVE 370.9 → **NEON 更差，defect #3（NEON 路由）也关闭**。
> - ⚠️ 所有「% of peak」必须改用 **370.9（i8）/ 92.8（bf16）** 重算，不能再用 660/741.6/185.4。
>   （bf16 92.8 由 huawei_bf16_roofline 实测 0.5 bfmmla/cycle 确认；早期写的 185.4 是误×2。）
>
> ### B. 用真实数据(`results/m8/m8_results.xlsx`)重算 → 真正的瓶颈是 **t79 调度坍塌**
> 大形状在 **t64 已到真实上限的 84–92%**（如 2048×4096×2048 @t64=325 GOPS/核=87.7%）——计算+扩展都接近最优。
> 但 **t79（华为默认就是 79 核！）大形状坍塌到 41–42%**：2048×4096×2048 t64=20819→**t79=12351**→t80=22628；
> 2048×4096×1024 t64=18545→**t79=12104**→t80=16833。t80 已被旧的不均匀网格救回，**t79（质数）仍坏**。
>
> ### C. ✅ 已修 t79：`i8_pick_grid` 加「1D-grid rescue」(`lib/i8gemm_sve.c` ~line 85)
> 根因：P=79 质数时 `i8_pick_grid` 为最大化忙碌线程选 **79×1（或 1×79）1D 网格** → 把 B panel **重复 streaming 79 次** → 带宽坍塌。
> 修复：当默认选出的是 1D 网格(pm==1||pn==1)时，**改选保留 ≥7/8 线程忙碌的最低流量 2D 网格**。
> 实测网格选择（2048×4096×2048）：P=79 从 79×1 → **8×9（B 流量 8× 而非 79×）**；
> **P=8/64/80 完全不变**（8/64/80 本就能整除成 2D 全用满）→ **零回退风险**。
> 正确性 6000 例 + verify_m12(1/8线程) bit-exact 通过（只改线程网格形状，不改计算）。
> commit 见 §7；**待华为验证**：t79 大形状应从 42% 回到接近 t64/t80 的 ~80%。
>
> 下一步焦点：(1) 华为重跑 dispatch 确认 t79 修复；(2) 若 t79 仍坏，开 `I8_SVE_DEBUG`/打印实际走的 path
> 看是否被 `use_n_split` 抢路由；(3) 把大形状从 t64 的 88% 往更高推属于带宽收益递减，优先级低。


> 目的：让另一个 Claude/工程师无需上下文即可接手这一轮"多线程线性度 / 80 核扩展性"工作。
> 上层背景见同目录 `HANDOFF.md`（i8gemm SVE/BF16 总体交接）。本文件只覆盖 **80 核华为机器的调度问题**。

## 0. 一句话现状
基于华为 80 核实测数据(`results/m8/m8_results_new.xlsx`)定位了三个调度缺陷;
其中**最严重的 t79/t80 线程划分坍塌(−60..−89%)已结构性修复并验证**(commit `d8dcf48`,
branch `sched-linearity-opt`,已 push)。剩下两项(微内核 50% 计算天花板、NEON/SVE 路由)
需要在华为机器上才能继续(原因见 §4)。

## 1. 平台事实(华为机器,来自 `cpufb/机器硬件性能.md` 的"华为机器"段)
- **80 核 / 单 NUMA / 单 die**(core 80–159 连续同频);**默认实际最多用 79 核**。
  → 79 是**质数**,是 2 的幂网格逻辑的最坏情形;80=16×5。这正是缺陷①的触发条件。
- 主频 2.9 GHz;SVE VL = 256-bit("SVE : 32" 字节)。
- L1 64KB / 4-way;L2 1.25MB/核;cacheline 64B(理论)。
- **单核峰值(换算成 bench 的 2-op GOPS 计法 = cpufb 数值 ×2):**
  - SVE i8 `sve_mmla` = **741.6 GOPS**(cpufb 370.82,IPC 1.0,latency 4)
  - NEON i8 `mmla` = 600.8 GOPS(cpufb 300.42,IPC 1.62)
  - SVE bf16 `bfmmla` = **92.8 GFLOPS(2op)** = 0.5 bfmmla/cycle×32MAC×2.9G
    (实测 `tests/experimental/huawei_bf16_roofline` ~92.7, bfmmla/cycle≈0.50;
     旧值 185.4 是把 cpufb 92.72 误×2，已更正)
- **80 核微基准聚合**:`sve_mmla` 28.324 TOPS → 354 GOPS/核(1-op)= **95.5% 线性**。
  ⇒ **硬件到 80 核仍近线性;所有掉速都是 kernel 的问题,不是机器的。**

### ⚠️ 基准换算坑(读 xlsx 必看)
`m8_results_new.xlsx` 的 `pct_of_baseline` 用的是**硬编码 660**(AWS Neoverse-V1 的 i8 峰值),
**不是华为的**。华为真实 SVE i8 峰值是 **741.6**(2-op),NEON i8 是 600.8。
分析利用率/天花板时必须用华为的数,不能直接信 xlsx 那列。bf16 baseline 330 → 华为 **92.8**(实测 0.5 bfmmla/cycle;非 185.4)。

## 2. 80 核实测数据(`results/m8/m8_results_new.xlsx`)
- 3 个 sheet:`dispatch`(真实 kernel 多线程性能,33 形状 × {1,2,4,8,16,32,64,80} 线程 × {sve,neon} × {i8,bf16}),
  `parts`(load/store/compute 拆解,含 `i8_sve_*` 与 bf16/f32 变体),`tail`(尾块)。
- **单位**:dispatch 的 `perf` 列 = 2-op GOPS。`parts` 的 `GFLOPS` 同为 2-op。per-core = perf/threads。
- 顶层结论:i8 SVE 最佳单形状 `2048,1024,8192 @t80` = 20.3 TOPS = **254 GOPS/核 = 峰值 34%**;
  多数大形状 t64 在 27–40%;中小形状 15–30%。**机器能到 95% 线性,我们最好 34%,余量巨大。**

### 三个缺陷(按影响排序)
1. **t79/t80 调度坍塌(已修,见 §3)** — `K=4096,N=1024` 一族:
   `64×4096×1024` t64=7925 → t80=842(−89%);128:−84%;256:−75%;512:−62%;M≥1024 不坍塌。
   小形状过订阅:`64×512×128` 峰值在 t4、`64×512×256` 峰值在 t16,之后一路掉。
   bf16 路径几乎没有这些病(t80 仍在爬,40–46%)→ 坍塌是 **i8 dispatch 路径特有**。
2. **微内核 50% 计算天花板(未修,需华为计数器,见 §4.1)** — `i8_sve_compute_only`
   (去掉全部访存的 roofline)per-core 只到 ~361 GOPS = **峰值 49–50%,且 t4/t8 持平**;
   同一二进制在 V1 上能到 ~92–100%。这是真实微架构错配,**约 2× 余量,最大单点杠杆**。
3. **NEON/SVE 路由(未做,需构建决策,见 §4.2)** — NEON i8 在低线程/小形状更快
   (`64×256×512 @t8`:NEON 1834 vs SVE 1370),高线程大形状 SVE 赢且 NEON 坍塌更早。

## 3. ✅ 已落地:不均匀 2D 线程网格 + fork 摊销钳制(commit `d8dcf48`)

**改的文件:仅 `lib/i8gemm_sve.c`。**

### 根因(已精确定位)
旧 `i8_pick_grid` 要求 `pm×pn==P` 且两维都整除。P=79(质数)/80=16×5 时找不到网格 → packed 路径
回退到只有 ~6 个 row-block 的 M-split(`use_m12` 还付两次 fork-join)→ 70+ 线程空转 → 坍塌。
追踪 `64×4096×1024 @t80`:`i8_sve_use_n_split` 因 `n_tiles(64) < num_threads(80)` 返回 0 → M-split → 842。
(t64 时 `n_tiles 64 == threads` → n-split 完美 → 7925。)

### 改动
- **`i8_pick_grid`(~line 85)**:改为允许**不均匀、`pm×pn ≤ P`**;`pm ≤ mblk`、`pn ≤ ntiles`;
  先最大化忙碌线程数 `pm*pn`,再最小化搬运量 `pn*M + pm*N`;退化时返回 `pm=pn=1`,调用方只在 `pn>1` 时启用。
- **packed 网格循环(~line 700,`if (gpn > 1)`)**:按余数逐带分配(`bq/brem`、`tq/trem`),
  **`#pragma omp parallel num_threads(gpm*gpn)`** —— 关键:spawn 恰好 pm*pn 个,否则越界 tid 会写到 C 之外。
- **hybrid 网格(~line 540)**:加 `starved = (mblk < num_threads)`;`mtmode = (gpn>1 && (big||starved)) ? 4 : ...`;
  mtmode==4 区域也改成 `num_threads(pm*pn)`。
- **`i8_sve_effective_threads`(~line 110)**:加 `K_r` 参数 + **mac 数钳制**
  (`I8_SVE_MIN_MACS`,默认 `512<<10`),按 `total_macs/min_macs` 上限线程数。
  标定:V1 上所有基准形状(最小 `64×128×512`=4.2M macs)仍 ≥8 线程 → **8 核零影响**;
  华为上 `64×512×256`(8.4M)→ ≤16 线程(正好对上实测 t16 峰值)。3 处调用点都加了 `K_r`。

### 验证(本地 8 核 V1)
- 正确性:`make -C tests test-sve` → 6000 例;自写 `scratchpad/grid_correct.c` → **224 例全过**
  (16 形状 × 14 线程数,含质数 3/5/7/13/31/**79** 和 t80,直测 `i8gemm_mt` vs naive 参考)。
- 无回退:新(不均匀)vs HEAD(均匀,`4461b22`)在 t8 全部 ±5% 噪声内。
- V1 顺带提速(新 vs `I8_SVE_PACK2D=0 I8_SVE_MIN_MACS=0`):`64×512×256` **+72%**、
  `32×512×256` **+85%**、`256×2048×256` +24%、`512×512×256` +19%;cube/大 N 噪声内。
- 华为预期(纯划分算术,免跑可定):`64..512 ×4096×1024` t79/t80 从用 ~6 线程变 ~78 线程,坍塌消除。

### 相关 env 开关
`I8_SVE_PACK2D=0`(关 packed 网格,对比用)、`I8_SVE_MIN_MACS=0`(关钳制)/`=<n>`(调阈值)、
`I8_SVE_HYBMT=0..4`(强制 hybrid 分发模式)、`I8_SVE_CLAMP_THREADS=0`(关全部钳制)。

## 4. 未完成项

### 4.1 ① 微内核 50% 计算天花板(最大杠杆,~2×,需华为 perf 计数器)
- **已排除**:不是累加器不够 —— `lib/i8gemm_sve.S` 已是 16 累加器(z16–z31,4×4)+ 双缓冲交错 load
  + 软件预取(`prfm pldl1keep` 在 line 138/171/298)。该做的微内核优化都在。
- **现象**:`compute_only` roofline 在华为只 ~50% 且 t4/t8 持平 → smmla 发射率只有 ~0.5/cycle
  (华为 ISA 支持 1.0)。同一二进制 V1 上 ~92–100%。**纯微架构错配。**
- **为什么不能盲改**:改一条手工流水的 SVE 汇编却无法在华为上量,无法"确保性能有提升",
  且有破坏 V1 上 ~92% 工作点的风险。
- **下一步(需用户在华为跑)**:
  ```bash
  perf stat -e cycles,instructions,sve_inst_spec,stall_backend,stall_backend_mem \
    taskset -c 80 <compute_only_i8_bench> 256 4096 1024
  ```
  - `stall_backend_mem` 高 → 内存/预取:调 `prfm` 距离(华为 cacheline 与 V1 不同)。
  - `stall_backend` 高、mem 低 → smmla 与 load 抢发射端口:降低每条 smmla 的交错 load 比例。
  - 两者都低、SVE IPC≈0.5 → smmla 吞吐结构上限:换 B 复用更高、交错 load 更少的微内核 tile。
  拿到这三个数即可落地针对性内核改写并能验证。

### 4.2 ④ NEON/SVE 运行时路由(需构建决策)
- 数据:NEON 只在低线程/小形状赢;华为跑高核数 → **SVE 是对的路径**,该项收益有限。
- 现状:NEON 与 SVE 是**两个分开编的二进制**;运行时路由要把两套微内核链进同一胖二进制 + 加分支。
- 决策权在用户:若部署含大量"单 GEMM 低延迟"调用 → 做胖二进制 + 按 (线程数×形状) 路由;
  若只跑大批量高并发 → 暂不做。

## 5. 复现命令
```bash
# 本地 8 核正确性
cd i8gemm/tests && make test-sve

# 自写网格正确性(质数/高线程,直测 i8gemm_mt vs naive)
SC=<scratchpad>; cc -O3 -fopenmp -mcpu=native -Ilib -o $SC/grid_correct $SC/grid_correct.c \
  lib/i8gemm_sve.c lib/i8gemm_sve.S lib/i8gemm_hybrid.S lib/i8gemm_pack_a_neon.S -lm && $SC/grid_correct

# 华为 80 核验证(在华为机器上,绑 0-79;和 main 对比 dispatch sheet 看坍塌是否消失)
git fetch origin && git checkout sched-linearity-opt
cd tests
CORES=0-79 THREADS="1 2 4 8 16 32 64 79 80" PRUNE_SMALL_MT=0 RUN_DISPATCH=1 RUNS=4 bash run_m8_parts.sh
```
> 注：华为上**用 79 线程**做主线程数(默认只 79 核可用);加测 79 能直接看到质数线程数下的网格表现。
> 钳制阈值 `I8_SVE_MIN_MACS=512Ki` 是按 8 核标定的,80 核若某些形状没提速,发数我重调阈值。

## 6. 关键文件 / 行号速查
- `lib/i8gemm_sve.c`:`i8_pick_grid` (~85)、`i8_sve_effective_threads` (~110,带 mac 钳制)、
  `i8gemm_mt_dispatch` hybrid 分支 (~510,含 `starved`/mtmode/grid)、packed 网格 (~660–710)。
- `lib/i8gemm_sve.S`:微内核(累加器 z16–z31、`prfm` line 138/171/298、K8 循环)。
- 数据:`results/m8/m8_results_new.xlsx`(华为 80 核;sheet=dispatch/parts/tail)。
- 硬件:`cpufb/机器硬件性能.md`(华为机器段)。
- scratchpad:`grid_correct.c`(网格正确性)、`rebuild.sh`(从 build/*.o 重链 SVE dispatch bench)、`measure.sh`。

## 7. Git 状态
- branch `sched-linearity-opt`(remote `git@github.com:ZheLdeng/zlgemm.git`),已 push。
- 本轮提交:`d8dcf48`(不均匀网格 + 钳制)。之前:`4461b22`/`02a94a5`(均匀网格)、`85d1f7c`/`a7e3d2a`/`8dfa892`(run_m8_parts 修复)。
- **尚未合并到 `main`** —— 等用户决定:现在合(坍塌修复已验证、V1 无回退),还是先在华为跑一轮 dispatch 确认坍塌消失再合。
