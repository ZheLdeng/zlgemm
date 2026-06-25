# i8gemm SVE/BF16 优化 — 工作交接文档 (HANDOFF)

> 目的：让另一个 Claude/工程师无需上下文即可接手。记录目标、平台、已做的改动、
> 验证方式、性能现状、关键发现，以及下一步。

> **2026-06-24 — 80 核华为机器调度工作另起一份专门交接：见
> [`huawei_80c_scheduling_progress_2026-06-24.md`](huawei_80c_scheduling_progress_2026-06-24.md)**。
> 内容：基于 `results/m8/m8_results_new.xlsx` 定位的三个调度缺陷；t79/t80 线程划分坍塌
> (−60..−89%) 已修并验证 (commit `d8dcf48`, branch `sched-linearity-opt`)；
> 微内核 50% 计算天花板与 NEON/SVE 路由两项待续(需华为机器)。

> **2026-06-25 — 小 M 多线程 + 独立 M-split lib 工作另起一份专门交接：见
> [`smallm_msplit_handoff_2026-06-25.md`](smallm_msplit_handoff_2026-06-25.md)**（差距分析见
> [`gap_and_plan_2026-06-25.md`](gap_and_plan_2026-06-25.md)）。内容：基于 ACL 系统对比，修复
> 最严重差距(小 M 浅/中 K 多线程 `8×512×512 @t8` 0.57→1.21× ACL)；新建只切 M 的
> `i8gemm_msplit` lib(小 M 用 split-K，1.5–2.6× over 朴素 M-split，7168 用例 bit-exact)；
> 残差全部收敛到微内核宽度(8×2VL vs ACL 6×4VL)，是 V1 专属、暂停等华为验证瓶颈。
> branch `i8-msplit-and-smallm-opt`(已 push)，commits `b326b41`/`60ba942`/`f6dc87b`。

## 0. 一句话现状
i8gemm 的 SVE i8 路径经过本轮优化后，在**小/中/偏斜形状**上已普遍追平或超过 ACL，
**bf16 在所有测试点领先 ACL 和 KleidiAI**；剩余差距是 **i8 中等立方体 (192–256³) 在 8 线程**
（本轮 2D blocking 后已大幅收窄，见下）。

> **2026-06-23 更新 — priority #2（中等立方体 @8 的 2D blocking）已大幅推进**：
> 在 hybrid（免 pack）多线程路径里把"只按 M 切 8 行块"改为**自适应 2D 分发**
> （`lib/i8gemm_sve.c` 的 `i8gemm_mt_dispatch` hybrid 分支，`M>8` 部分）：
> ①能整除时用 **2D 线程网格 pm×pn**（既均衡又减少 A/B 冗余 streaming，聚合流量 ~pn·|A|+pm·|B|）；
> ②`mblk` 不被线程数整除时用 **8行×n_tile 的 2D collapse**（消除 M-split 末块掉队，最高 +23%）；
> ③否则保持 M-split；④`M<128 或 N<128` 一律 M-split（网格/2D 在小形状上有开销，64–112 实测会回退）。
> 结果（vs ACL，8 线程，runs=8）：160³ 0.86→**0.99**、192³ 0.94→**0.98**、200³ 0.80→**0.95**、
> 208³ 0.84→**0.95**、224³ 0.82→**0.94**、240³ 0.87→**0.92**、256³ 0.84→**0.89**；
> 小形状 (64–112) 与其它线程数 (t2/t4/t6) 无回退（t4/t6 反而 +8~11%），正确性 6000 例通过。
> CSV: `results/m8/i8_medium_cube_2d_blocking_2026-06-23.csv`。
> 剩余差距集中在 256³ (0.89)，判断为 A+B 的 LLC 聚合带宽 / prefetch 瓶颈（见 §6.2）。
> 环境变量 `I8_SVE_HYBMT` 可强制分发模式做对比实验（0=M-split 1=N-split 2=2D-collapse 3=均衡带 4=2D-grid）。

> **2026-06-23 更新 — 预排上限之谜已解决（priority #1 已完成）**：实测本仓库的预排 in-L1 微内核上限，
> **SVE `i8gemm_k_nld_m12` (12×16) = 89.7%，反而高于 NEON `i8gemm_k_reo_ld` (8×8 K16) = 78.4%**
> （CSV: `results/m8/prepacked_ceiling_neon_vs_sve.csv`）。即 **“SVE 结构性低于 NEON / 应改走 NEON”
> 的假设被推翻**——本仓库里 SVE 微内核才是更优的那个，不应让 i8 dispatch 在 V1 上改走 NEON。
> 用户所称 NEON “>95%” 并非指本仓库这个 8×8 reo_ld 内核（可能是 ACL 自身的或更宽的内核）。
> 因此剩余工作聚焦在 priority #2（中等立方体 @8 线程的 2D cache-blocking），而非更换微内核。

## 1. 项目与平台
- 代码：`/home/ubuntu/zhelang/i8gemm/`（独立 git 仓库，**改动全部未提交**）。
- 参考：ACL = `/home/ubuntu/zhelang/ComputeLibrary/`（已 build 在 `build-codex/`），
  KleidiAI = `/home/ubuntu/zhelang/kleidiai/`。
- 平台：Neoverse-V1，aarch64，**SVE（非 SVE2）**，VL=256bit(32B)，i8mm+bf16，8 核，2.6GHz。
- **峰值**（来自 `/home/ubuntu/zhelang/cpufb/output.log`，单核，2-ops/MAC 口径）：
  - i8 smmla：**661 GOPS/核**（cpufb 报 330.6 GOPS(MAC) ×2，IPC≈2）
  - bf16 bfmmla：**331 GFLOPS/核**（165.5 ×2）
  - 关键加载吞吐：`ld1rqb`=`ld1b`=**2 loads/cycle**；NEON `ldr q`=**3/cycle**（`ldp` 还能 2 reg/指令）
  - L1 64KB(4-way)，L2 1MB/核，cacheline 64B。
- benchmark harness：`tests/bench_dispatch_types.c`（用 `-DBENCH_SVE` 编译走 SVE）。
  - 用法：`<bin> impl bf16|i8|both M K N reps warmup runs threads`，输出 CSV，perf 在第 11 列(0idx=10)。
  - `pct_base` 列恰好 = 占单核峰值的百分比（默认 baseline 660/330 = 单核峰值）。

## 2. 构建与验证
```bash
make -C lib            # 生成 libi8gemm_sve.{a,so}
make -C tests test-sve # SVE 正确性：应输出 "SVE correctness OK: 6000 wrapper+dispatch cases"
```
- 额外对拍程序（实验目录）：
  - `tests/experimental/verify_m12.c` —— i8 i32 全路径对拍 naive int32 参考（1/8 线程 ALL OK）。
  - `tests/experimental/verify_hybrid.c` —— hybrid kernel 对拍。
- 所有改动均 **bit-exact**（i32 与参考逐元素相等；f32/bias 在 6000 用例容差内）。

### ⚠️ 工具坑（重要）
- 反复编译 `prepacked_ceiling.c`（含大文件 `bf16gemm_sve.c`）在本环境**经常 120s 超时**——
  机器负载其实很低，是工具/编译时间问题。**对策**：先 `cc -c` 预编译成 `.o`（`build/_i8c.o`
  `_i8s.o` `_pa.o` `_bf.o` `_bfs.o`），再链接；或只链接 i8 文件不带 bf16。
- 后台残留进程会拖慢测量，跑完用 `pkill -9 -f build/pp` 清理。
- 一次只跑一条测量命令；多条 `for` 循环的 stdout 经常被终端回显吞掉，单独跑更可靠。

## 3. 本轮已落地的改动（按文件）
**`lib/i8gemm_sve.S`**（i8 SVE 汇编）
1. **修复 96³ 崩溃 + m12 i32 数值错误 bug（关键正确性修复）**：
   `BUILD_I32_OFFSET_BASE` 原用 `z0` 当临时寄存器（`dup z0.s,w12`），与 m12 存储调用
   `BUILD_I32_OFFSET_BASE z0`（out=z0）冲突 → 偏移算成 `2*ldc` → 越界写 C（崩溃）且
   **所有 m12 i32 形状结果错误**（K_r≥64,M≥12,n_tiles>1，6000 用例没覆盖到所以之前没发现）。
   修复：临时寄存器改 `z2`。
2. **存储反交织（scatter→连续）**：`DEINT_STORE_PAIR`/`DEINT_QUADS` 宏。
   实测 SVE scatter `st1w` 比连续 `st1w` **慢 7.2×**。用 `trn1/trn2.d` 分离奇偶行+归组列，
   `ext` 取高 quad（V1 无 SVE2 的 ZIP1.q），全部连续 `st1w`，`p2=VL4` 谓词。
   覆盖：i32 8 行(`STORE_C_I32`)、i32 m12、f32(`STORE_C_F32`)、f32+bias(`STORE_C_F32_BIAS`)。
3. **软件预取**：m12 body + `MMLA_CUR_LDNEXT`/`NEXT_LDCUR` 加 `prfm pldl1keep,[x6,#512]` 等。
   实测 **V1 上中性**（硬件预取器已饱和线性流），保留（无害、可移植）。

**`lib/i8gemm_hybrid.S`（新文件）** —— 小形状 hybrid 无 pack 微内核 `i8gemm_k_hybrid`：
直接从行主序读 A（branch-free `ldr d`+`ld1`+`dup`，无独立 pack pass），8×16 tile，连续反交织存储。

**`lib/i8gemm_sve.c`**（dispatch）
- 线程本地缓存 A-reorder scratch（`g_a_poolA/B` + `i8_scratch()`），**去掉每次调用的
  `aligned_alloc`/`free`**；pool prepare 函数改为接收 buffer 参数。
- `i8_use_hybrid_for_shape(M,K_r,N_r,num_threads)` 路由：
  - 单线程：`N_r≤256 && ((K_r≤128&&M≤64) || (M≤16&&K_r≤256))`
  - **多线程 (nt≥2)：`K_r≤256 && M≤256`（N 不限）** —— 因为 packed 路径有独立 A-pack OpenMP
    区，fork 开销在小工作量时主导；hybrid 无 pack、扩展好得多。
  - 路由执行：nt≤1 单调用；M≤8 多线程按 N-tile 切分（N-split）；否则按 8 行 M-split。
  - 可用环境变量 `I8_SVE_HYBRID=0/1` 覆盖。

**`lib/Makefile` / `tests/Makefile` / `tests/run_three_lib_grid.py`**：加入 `i8gemm_hybrid.S`。

**未改动**：`bf16gemm_sve.{c,S}`（bf16 本就领先，本轮没动；早期试过 bf16 主 kernel 交错调度，
中性，已回退到 HEAD）。

### 试过但放弃的（不要重复走）
- **bf16 主 kernel 交错软件流水**：端到端中性，回退。
- **i8 main kernel K16 展开 + 摊销指针推进**：隔离 91%，但端到端中性/略负（瓶颈是 store+overhead 不是内循环），回退。
- **fused pack+compute（每线程就地打包再用快 packed kernel）**：256³@8=1812 < hybrid 1869，
  128³@8=839 < hybrid 1295，**更差**，已删除。结论：hybrid（完全不 pack）才是这些形状最优。

## 4. 性能现状（i8，vs ACL，单位 GOPS）
单核峰值 661。利用率 = 占 (线程数×661)。
- **单线程利用率 ~79–87%**（大 shape 512³/2048big 到 85–87%）。
- **8 线程利用率 ~39–58%**（占 8×661；受共享内存带宽限制，绝对值高但占比降）。扩展比大 shape ~5×。

**对 ACL 战绩**：
- 赢/平：tiny 立方 16–64（1t/4t）、**96/128 立方（1t/4t/8t，128³@8: 1236 vs 956）**、
  384/512 立方（1t；8t 约平 3–5% 内）、小 M（8×256 全线程）、大 N 偏斜多线程
  （64×64×512@8: 1110 vs 1058）、小 K 偏斜（256×16×256 比优化前 +106%）。
- ACL 仍领先：**i8 192–256 立方 @8 线程**——经本轮 2D blocking 已收窄到 192³ −2%、256³ −11%
  （之前 192³ −6%、256³ −15%；详见 §0 更新与 §6.2）；单线程 64–256 立方（中 K）。
- **bf16：所有测试点领先 ACL 和 KleidiAI。**

> **2026-06-23 重新核实（runs=4 best-of，CSV: `results/m8/i8_medium_cube_gap_vs_acl_2026-06-23.csv`）**：
> 192³@8 −7%、224³@8 −18%、256³@8 −17%（与历史一致，无回退）。
> **256³ 线程扫描** SVE/ACL：t1=0.95、t2=0.88、t4=0.83、t8=0.86 —— **差距随线程数变大**
> （单核微内核没问题，是多线程 L2/LLC 带宽 + cache-blocking 的差距）。SVE 8 核扩展 4.25×、ACL 4.69×。
>
> ### ⚠️⚠️ 测量方法坑（新增，极重要）
> **本机必须用 `runs>=4`（bench 取 best-of-runs）**。`runs=1` 会偶发“冷/被抢占”假崩溃：
> 256³@8 用 runs=1 读到 **121 GOPS**（真实 ~1900），192³ 读到 53。一开始误以为是回退，其实是噪声。
> 背景有 `claude`/`node`/`codex` 进程会瞬时抢核。**永远 runs>=4，且一次只跑一条 8 线程测量**。

关键 before/after（scatter→现在，GOPS）：256×16×256@1 64→131(+106%)；64³@8 92→360；
128³@8 525→1236；64×64×512@8 446→1110；512³ 526(1t,80%)；2048×4096×8192@8 2727→2844。

## 5. 关键发现（决定下一步方向）
1. **纯 compute = 100%**（`compute_ceiling`），**预排 in-L1 kernel = 84–87%**（m12/8 行）。
   差距来自加载 + 循环/store 开销，不是指令选择。
2. **SVE 加载吞吐确实低于 NEON**（`ldcost` 实测）：`ld1rqb`/`ld1b`=2/cyc，NEON `ldr q`=3/cyc。
   SVE-256 当 2×128 用时，smmla 的 A 操作数**必须用 `ld1rqb` 复制**到两个 segment；NEON 单 segment
   用 `ldr q` 直接读 + `ldp` 配对。**但这并未转化成更高的端到端预排上限**——见下。
   - ✅ **2026-06-23 已确认（用 `i8gemm_k_reo_ld` 读预排 A + L1 常驻形状）**：
     - NEON `i8gemm_k_reo_ld` (8×8, K16, cached path first_n=0)：**最高 78.4%**（M8 N16 K2048）。
     - SVE `i8gemm_k_nld_m12` (12×16)：**最高 89.7%**（M12 N16 K2048）。
     - 即**本仓库的 SVE 预排内核比 NEON 预排内核高 ~11pp**。8×8 NEON 理论上是 compute-bound
       （K16 内 32 smmla / 16 ldr-q，加载不该是瓶颈），实测只有 78%，差距来自循环/累加器调度，
       而非加载吞吐。结论：**load 吞吐的劣势并未让 SVE 落后；反而 SVE 12×16 更宽的 tile 摊薄了开销。**
     - 旧 `neon_ceiling.c` 探针（调 `i8gemm_k_ld`，每次内部重排 A）测得 30–63% 是被 A-pack 拖累的
       伪低值，已被 `neon_reo_ceiling.c`（调 `reo_ld`）取代（旧探针已删）。探针二进制见 §7。
3. 多线程中等形状之前扩展差的根因 = packed 路径的**独立 A-pack OpenMP 区**的 fork 开销；
   改走 hybrid（单 region、无 pack）后大幅改善。

## 6. 下一步（按优先级）
1. ~~**确认 NEON 预排 kernel 真实上限**~~ ✅ **已完成 (2026-06-23)**：NEON reo_ld=78.4% < SVE m12=89.7%。
   结论 = **不要改走 NEON**（SVE 微内核本就更优）。SVE 单线程端到端已 79–87%，离 90% 内核上限很近，
   单线程已接近最优，无需再优化微内核。详见 §5.2 与 `results/m8/prepacked_ceiling_neon_vs_sve.csv`。
2. ~~**【当前主攻】i8 192–256 立方 @8**~~ 🟡 **大幅推进 (2026-06-23)**：已确诊为**多线程 L2/LLC 带宽 +
   cache-blocking + 负载均衡**问题（**不是微内核问题**：256³ 单核 SVE/ACL=0.95，差距随线程数拉大到 0.84）。
   **已做**：保留更优的 SVE hybrid 微内核，把 `i8gemm_mt_dispatch` 的 hybrid `M>8` 分支从"纯 8 行 M-split"
   改成**自适应 2D 分发**（细节见 §0 更新与 `lib/i8gemm_sve.c`）。结果：192–256³ @8 从 0.80–0.94 提到
   **0.89–0.98**，最差点 200³ 0.80→0.95，正确性 6000 例通过，小形状/其它线程数无回退。
   CSV: `results/m8/i8_medium_cube_2d_blocking_2026-06-23.csv`。
   **关键洞察**：当年怀疑的"B 重复 streaming"只在 `mblk` 不被线程数整除时才是主因（其实是末块掉队的负载不均，
   8 行块粒度太粗）；能整除的 192/256 用 2D 网格 (pm×pn) 减少 A+B 聚合带宽再拿到几个点。
   **剩余**：256³ 仍 0.89（A+B 的 LLC 聚合带宽 / ACL `sve_hybrid_s8s32_mmla_6x4VL` 的 prefetch/K-streaming
   更优）。下一步可试：(a) 显式 K-blocking 让 A/B panel 更常驻；(b) 软件 prefetch B 下一 panel；
   (c) 给 hybrid 内核做更宽的 N tile（贴近 ACL 4VL）。
   ⚠️ 注意：之前的 **`i8gemm_m16n4` “2d mode” 失败了**（用了较差的 NEON 微内核，普遍输给 auto，
   见 `results/m8/i8_neon_aclstyle_2d_probe.csv`）；本轮正确做法已验证 = **复用 SVE 微内核只改外层分发**。
3. 可选：把 hybrid 扩展到 f32/bias（目前 hybrid 只做 i32）。
4. **收尾提交**：所有改动未提交，bit-exact 无回退，可整理成若干 commit
   （bug 修复 / 反交织 / 缓存 scratch / hybrid+路由 / 预取）。

## 7. 有用的文件/探针
- 主报告（详尽，按时间顺序记录每步实验 + step-by-step 优化历程）：
  `results/reports/gap_and_plan_2026-06-25.md`（附录 A 整合了已删除旧报告的对比数据）。
- 三方对比 CSV：`results/m8/three_lib_compare_session.csv`。
- 对比脚本：`tests/run_small_grid.py`（square/ksmall/msmall/nsmall 形状集，调三方 bench）。
- 微基准（`tests/experimental/`）：
  - `compute_ceiling.{c,S}` —— 纯 compute 与 compute+load 上限（i8/bf16）。
  - `prepacked_ceiling.c` —— 预排 in-L1 kernel 上限（i8 k_nld/m12、bf16 m12）。
  - `ldcost.{c,S}` —— `ld1rqb`/`ld1b`/`ldr q` 吞吐对比（→ 2/2/3 per cycle）。
  - `store_cost.{c,S}` —— scatter vs 连续 st1w（→ scatter 慢 7.2×）。
  - `verify_m12.c` / `verify_hybrid.c` —— 正确性对拍。
  - `repro96.c` —— 96³ 越界复现（guard page，已修）。
  - `neon_reo_ceiling.c` —— ✅ NEON 预排上限探针（调 `i8gemm_k_reo_ld`，cached path）→ **78.4%**。
    链接：`cc -O3 -mcpu=native -fopenmp -Ilib neon_reo_ceiling.c <i8gemm_k.o> <pack_a.o> <shim:i8_pack_B> -lm`
    （不能直接链 libi8gemm_sve.a：i8gemm_k.S 与 i8gemm_sve.S 的 i8gemm_k_ld* 符号重名）。
  - `sve_m12_ceiling.c` —— ✅ SVE 预排上限探针（调 `i8gemm_k_nld_m12`）→ **89.7%**。链 libi8gemm_sve.a 即可。
  - 结果汇总：`results/m8/prepacked_ceiling_neon_vs_sve.csv`。
- 关键结果 CSV（2026-06-23）：
  - `results/m8/prepacked_ceiling_neon_vs_sve.csv` —— NEON vs SVE 预排上限（78.4% vs 89.7%）。
  - `results/m8/i8_medium_cube_gap_vs_acl_2026-06-23.csv` —— 192/224/256³ @8 与 256³ 线程扫描（优化前基线）。
  - `results/m8/i8_medium_cube_2d_blocking_2026-06-23.csv` —— ✅ 2D 自适应分发前后 vs ACL（128–256³ @8，
    含线程扫描与非方阵抽查）；这是 priority #2 本轮成果的数据。
- 三方 bench 二进制：`tests/build/bench_dispatch_i8gemm_sve`（当前优化态）、`bench_acl_dispatch`、
  `bench_kleidiai_dispatch`、`bench_deint`（反交织但无 hybrid 路由，可作对比基线）。
  - ⚠️ 重建 `bench_dispatch_i8gemm_sve` 需带 m16n4 对象（bench 引用 `i8gemm_mt_dispatch_m16n4`）：
    `cc -O3 -mcpu=native -fopenmp -Ilib -DBENCH_SVE bench_dispatch_types.c <_bf _bfs _i8c _i8s _hyb _pa _m16c _m16s>.o -lm`。
    bf16gemm_sve.c 编译慢（易 120s 超时），先 `cc -c` 预编译成 .o（见 §2 工具坑）。
