# 小 M 调度 + M-split lib 优化 — 交接 (2026-06-25)

> 目的：让另一个工程师/Claude 无上下文接手这一轮"小 M 多线程 + 独立 M-split lib"工作。
> 平台：**AWS Neoverse-V1，8 核**（不是华为 80 核；华为专属调度见
> `huawei_80c_scheduling_progress_2026-06-24.md`）。差距分析与计划见同目录
> `gap_and_plan_2026-06-25.md`。代码已在 branch **`i8-msplit-and-smallm-opt`**（已 push 到
> `origin`，3 个 commit，未合并 main）。

## 0. 一句话现状
基于华为机器对手 ACL 的系统对比，修复了**最严重的差距**（小 M 浅/中 K 多线程坍塌，
`8×512×512 @t8` 从 0.57× ACL 升到 **1.21×** 并恢复线性扩展），并新建了一个**只切 M 的
独立 lib**（小 M 用 split-K）。剩余差距全部收敛到**微内核宽度**这一个根因，是 V1 专属、
高风险、且无法在华为验证的深水区——**已暂停，等华为侧验证瓶颈后再定**（见 §5 决策点）。

## 1. 这轮改了什么（按 commit）

### commit `b326b41` — 新 lib `i8gemm_msplit`（只切 M + 小 M split-K）
- 新文件：`lib/i8gemm_msplit.{c,h}`、`lib/i8gemm_msplit_k.S`；已接入 `lib/Makefile`、
  `tests/Makefile`（target `test-msplit` / `bench-msplit`）。
- 设计：与主 `i8gemm_sve` 的"可切 M 和 N"调度**平行**，这个 lib **只按 M 切**输出
  （从不细分 N 列）。
  - **M 够大**（`mblk=ceil(M/8) >= P`）：连续 8 行对齐的 M-band，每个 band 调用现有最优
    单线程 dispatch（`i8gemm_mt_dispatch(...,1)`），继承 packed/m12/hybrid 自动选择。
  - **小 M**（M-band 填不满线程）：**split-K**。把 K 归约切成 `ks` 段，worker 网格
    `(mg × ks)` 填满 P 线程，每线程算 `(M-band × 全 N × K-slice)` 的 partial，**单个
    parallel 区内**做 barrier 后全线程并行归约求和。
  - 为什么是 split-K 而不是更细的 M 切：`i8gemm_k_ld1/ld2/ld4`（1/2/4 行）其实都跑满 8 行
    smmla 机器（见 `i8gemm_sve.S` 的 `GEMM_BODY` 宏，所有变体同体），小于 8 行就**浪费算力**；
    split-K 保持 8 行 tile 满效率，同时让所有线程有活干。
- 新微内核 `i8gemm_msk` = `i8gemm_k_hybrid` 但 **B 的 N-panel byte stride 从 `params.ldb`
  读**（hybrid 是写死的 `K*n_tile`），这样一个 K-slice 能在一次调用里扫完整 N，而 B_reo
  仍用完整 K 的布局。plain M-split 时传 `ldb=K*n_tile, k=K` 即退化回 hybrid 行为。
- **正确性：7168/7168 bit-exact**（M=1..256，K=16..1024，N 低至 8，含质数线程 3/5/7/13）。
- **关键性能 bug 已修**：最初 compute / reduce 用两个独立 `parallel` 区，每次调用 team-size
  在 8↔mg 之间抖动 → `16×512×512 @t8` 只有 56 GOPS。改成**单 parallel 区**（compute +
  barrier + 全线程并行归约）后恢复正常。
- 调试 env：`I8MS_NOSPLITK=1`（强制 ks=1，即朴素 M-split 基线）、`I8MS_KS=<n>`（钉死 K-slice 数）。

### commit `60ba942` — 主 lib 小 M 路由 + 路径相关 fork clamp（修 gap #1）
- 文件：仅 `lib/i8gemm_sve.c`。
- **(1) 小 M 路由**：`i8_use_hybrid_for_shape` 多线程分支加 `M<=16 && K_r<=1024 → hybrid`。
  根因：M≤16 时只有 ≤2 个 8 行块，纯 M-split 填不满线程，packed 路径还要额外付 A-pack fork；
  hybrid 把 L1 常驻的 B panel 按 N 切给所有线程、单次廉价 fork。**M>=24 仍走 packed**
  （它的 2D grid + m12 kernel 能填满线程且更优，24=2×12 尤其适配 m12）。
- **(2) 路径相关 clamp**：`i8_sve_effective_threads` 增加 `min_macs_override` 参数。hybrid
  路径传 `64Ki`（单次廉价 fork，允许小形状在少核机上扩到满宽），packed 路径传 `-1` 用
  `512Ki` 默认（**保护华为 80 核**不被过订阅坍塌）。路由决策用**未 clamp 的请求线程数**，
  避免先 clamp 再判 hybrid。f32/bias 两个调用点传 `-1`（行为不变）。
- 同文件还**顺带提交了之前未提交的 t79 1D-grid rescue**（`i8_pick_grid`，待华为验证）。
- 结果（V1, runs≥4 vs ACL）：`8×512×512 @t8` 0.57→**1.21**、@t4 0.68→0.94；`16×512×512 @t8`
  →0.97。M>=24、大形状、小 M 深 K 均无回退（仍 1.1–1.8×）。6000 正确性用例通过。

### commit `f6dc87b` — 稳定残差表 + 根因报告
- `gap_and_plan_2026-06-25.md` 追加 round-1 结果 + runs=8 稳定残差表 + 根因 + 建议。
- `tests/gap_stable.sh`：runs=8 稳定复测残差簇的脚本。

## 2. 构建与测试命令
```bash
# 库
make -C i8gemm/lib sve

# 正确性
make -C i8gemm/tests test-sve       # 主 lib：6000 cases
make -C i8gemm/tests test-msplit     # msplit lib：7168 cases

# 差距扫描 vs ACL（写 results/m8/gap_sweep_sve_vs_acl.csv；runs=4，注意噪声）
cd i8gemm/tests && python3 gap_sweep.py
# 稳定复测残差簇（runs=8）
cd i8gemm/tests && bash gap_stable.sh

# msplit vs 主 lib（同一 prepacked B，同一计时口径）
#   bench_msplit M K N reps warmup runs threads  -> "M,K,N,t,main_sve,msplit,ratio"
cd i8gemm/tests
GOMP_CPU_AFFINITY=0-7 OMP_PLACES=cores OMP_PROC_BIND=close build/bench_msplit 8 2048 2048 30 2 4 8
# split-K 的价值（vs 朴素 M-split-only 基线）
I8MS_NOSPLITK=1 build/bench_msplit 8 2048 2048 30 2 4 8   # 基线
build/bench_msplit 8 2048 2048 30 2 4 8                     # split-K
```
> 测量纪律（本机踩过的坑）：**runs>=4**（bench 取 best-of-runs），背景有 claude/node 进程会瞬时
> 抢核造成假崩溃；多条命令的 stdout 会被终端回显吞掉，**写文件再 cat** 或一次跑一条。
> 跑完 `pkill -9 -f bench_` 清理残留。

## 3. 性能现状（V1 8 核）
- **gap #1（已修）**：`8×512×512` t8 1.21×、t4 0.94×；之前 0.57/0.68。
- **msplit lib**：M-split-only 范式内 split-K 比朴素 M-split **1.5–2.6×**（小 M 高线程）；
  vs 主 lib（可切 N）在极小内存受限形状偏低（N-split 把 B 只流一遍，split-K 付归约 + 私有
  buffer 流量），在 M-banding 能填满线程处持平/略胜（`64×512 t8` 1.03、`128×1024 t8` 1.06）。
- **稳定残差 vs ACL（runs=8，全在 0.74–0.98）**：
  ```
  单线程:   128³ 0.74  32×256×256 0.82  16×128×128 0.84  256³ 0.95
  K256N256 t4: 64×256 0.86  256³ 0.88  128×256 0.89  32×256 0.92  (t8 ~持平，grid 模式)
  N512 t8:  64×512 0.87  128×512 0.92  256×512 0.95  512³ 0.98
  ```

## 4. 残差根因（已确诊）
**全部** overhead/issue-bound 的中小形状，根因是**微内核宽度**：我们 **8×2VL（8 行×16 列）**
vs ACL `sve_hybrid_s8s32_mmla_6x4VL`（**6 行×32 列**）。更宽 N tile 把 N-tile 迭代数减半
（连带每 tile 的流水填充/排空 + A 广播开销），K/N 小时 ACL 摊销更好。调度已接近最优
（N-split/M-split/grid 三模式都测了，K=256 立方体 grid 模式胜）。

## 5. ⏸️ 决策点（暂停在此，等华为）
唯一剩下的杠杆 = **写一个 4VL 宽微内核**。暂停原因：
1. **V1 专属**：华为目标机 smmla = **1 条/cycle（V1 的一半，见 roofline 报告）**，中小立方体在
   华为上是**被 smmla 算力卡住**，不是 issue/overhead → V1 调优的宽内核在华为可能零收益。
   这轮调度优化（gap #1、msplit split-K）是**跨硬件可移植**的；微内核宽度差距不是。
2. 高风险 asm（6×4VL 几乎占满 z 寄存器堆），且无法在华为验证。

**华为侧待办（决定是否做 4VL 内核的前提）**：在华为单核跑 128³ 单线程，看 smmla 发射率
是否打满 1/cycle：
```bash
perf stat -e cycles,instructions,sve_inst_spec,stall_backend,stall_backend_mem \
  taskset -c 80 <128^3 single-thread i8 bench>
```
- 若 smmla≈1/cycle 且 stall_backend 高 → **compute-bound，4VL 内核没用**，不要做。
- 若 stall 低、issue 有余量 → 才考虑 4VL 宽内核（同时确认华为上确实存在该差距）。

## 6. 关键文件/行号
- `lib/i8gemm_sve.c`：`i8_use_hybrid_for_shape`（小 M 路由，~440）、`i8_sve_effective_threads`
  （`min_macs_override`，~135）、`i8gemm_mt_dispatch`（路由/clamp 顺序，~560）、`i8_pick_grid`
  （t79 rescue，~105）。
- `lib/i8gemm_msplit.c`：`msp_pick_grid`（mg×ks 网格 + env 开关）、`i8gemm_msplit_dispatch`
  （M-split / split-K 双路径 + 单区归约）。
- `lib/i8gemm_msplit_k.S`：`i8gemm_msk`（ldb-strided hybrid）。
- 数据/脚本：`results/m8/gap_sweep_sve_vs_acl.csv`、`tests/gap_sweep.py`、`tests/gap_stable.sh`、
  `tests/bench_msplit.c`、`tests/test_msplit_correctness.c`。
- 分析：`results/reports/gap_and_plan_2026-06-25.md`。

## 7. Git
- branch `i8-msplit-and-smallm-opt`（已 push `origin`），commits `b326b41` / `60ba942` /
  `f6dc87b`。基于 `a235a51`（main）。**未合并 main** —— 等华为确认 gap #1 修复无回退 +
  决定 4VL 内核方向后再合。
