# bf16 优化 — 从 i8 结论的迁移分析 (2026-06-25)

> 任务：把 i8 这轮的调度优化按数据推断迁移到 bf16。结论：**bf16 的调度器与 i8 不同，
> i8 最大的那个 win（窄 N 修复）不适用；其余迁移需华为 bf16 数据。** 不做投机性改动。

## 1. i8 各项优化能否迁移到 bf16
i8 的 bf16 单核/多核口径：peak = **185.4 GFLOPS/核(2-op)**（= i8 370.9 的一半，bfmmla 半吞吐）。

| i8 优化 | 能否迁移 bf16 | 依据 |
|---|---|---|
| 窄 N：2D-grid 加 n_tiles 门控（+51–97%） | **❌ 不适用** | i8 的病在 `pick_grid` 选 tall-pm/thin-pn 把 N 切成细列条；**bf16 的 2D 是 (m_unit×n_tile) collapse，每个 work item 处理一个完整 n_tile，不切细条**。V1 实测窄 N（n_tiles=2/3/4）bf16 default ≈ 强制 M-split（651 vs 641、881 vs 876…），**无病**。（曾出现 1 个 140 的离群点，复测确认是 8 核过订阅的瞬时抖动。）|
| t79 质数线程坍塌（1D-grid rescue） | **❌ 不适用** | i8 病根是 `i8_pick_grid` 质数 P 退化成 1D；bf16 不用 pick_grid。华为进度文档也记录 bf16 路径无 t79 病（t80 仍在爬）。|
| 小 M 路由 hybrid / GEMV(M≤4) 路由 | **🔶 待华为数据** | bf16 有 no-reorder 路径但门控更严（`M<=8 && n_tiles<=2`）。i8 的 GEMV win 来自 packed clamp 把 M=1 饿死；bf16 的 `clamp_threads_for_split` 同样把 n-split 限到 `(n_tiles+1)/2`、m-split 限到 `(m_blocks+1)/2`——**GEMV 可能同样被饿死**，但需华为高核数据确认（V1 8 核测不到，2D/clamp 路径需 ≥10 线程）。|
| 线程拐点 advisory | 🔶 可加 bf16 版 | 同理，需华为 knee 数据校准。|

## 2. 为什么不做投机性改动
- bf16 报告口径**在所有测试点已领先 ACL 和 KleidiAI**（HANDOFF）。
- bf16 调度器（collapse + 自己的 n-split/2d/nblock/clamp）与 i8 结构不同，i8 的具体修复多数不对应。
- **V1 8 核测不到 bf16 的多核病**（2D 需 ≥10 线程、clamp 在高核才咬）——盲改一个已经赢、且无数据支撑的路径会冒回退风险。
- 教训：差点因 1 个离群点（2048×16384×24=140）误加窄 N 门控，复测证明是抖动 → 坚持"先测后改"。

## 3. 交付：bf16 测量工具（华为上跑，数据驱动再改）
- `tests/huawei_fine_sweep.sh` 加 `DT` 参数：`DT=bf16 bash tests/huawei_fine_sweep.sh` 出 bf16 的 247 形状利用率（peak 自动用 185.4）。
- `tests/huawei_bf16_probe.sh`：对窄 N / 小 M-GEMV / 中 cube / 大形状，比较 default vs `BF16_SVE_SPLIT={m,n,2d}`，找 bf16 的便宜调度修复。

### 华为运行
```bash
cd ~/zhelang/i8gemm && git pull
CORES=0-79 DT=bf16 bash tests/huawei_fine_sweep.sh 2>&1 | tee bf16_fine_sweep.log   # 利用率全景
CORES=0-79 bash tests/huawei_bf16_probe.sh 2>&1 | tee bf16_probe.log                 # 调度模式对比
```
判读：
- bf16_probe 若某 split 模式在小 M/GEMV 大幅领先 default → 那就是 bf16 版的"GEMV 路由/clamp 放宽"，我据数据落地（对应 i8 的 #4）。
- fine_sweep(bf16) 给出 bf16 各形状利用率范围 + 拐点线程，对照 i8 找 bf16 特有弱点。

把这两个 log 发我，我按 i8 同样的数据驱动流程优化 bf16（只改数据支持的项）。

---

## 4. 华为 bf16 实测结果（更新：窄 N 修复**确实迁移**）
跑了 `huawei_bf16_probe.sh` + `DT=bf16 huawei_fine_sweep.sh`（CORES=0-79, RUNS=4）。

### ✅ 窄 N（n_tiles=2）：2D collapse 确实有病 → 已修
之前 V1 8 核（过订阅）误判为"无病"；**真华为数据推翻**：
- `2048×16384×24`（N=24→n_tiles=2）：default(2D)=795 vs M-split=**1465 @t64（+85%）**、t48 +58%、t32 +36%。
- `2048×4096×64`（n_tiles=4）：default 2D=1865 最优（2D 没病）。
→ **修复**：`bf16_use_2d_split` 加 `n_tiles >= 4` 门控（窄 N 退回 M-split）。这是 i8 `n_tiles>=8` 修复的 bf16 版（bf16 阈值更低，因 collaple 比 i8 的 pick_grid 更耐窄 N）。正确性验证：`check_bf16_split` 8 形状 × {default,m,n,2d} 全 OK。**待华为复测确认 default 吃到 +36–85%**。

### 🔶 其余（default 已够好或属微内核，不改调度）
- 小 M / GEMV（`8×4096×1024`、`1×4096×4096`）：default(走 n-split/2d) 已好，forced-m 才崩（M-split 无法并行）→ **i8 的 GEMV clamp 饿死问题不在 bf16**（bf16 默认就路由到 n-split/2d），不改。
- 256³ @t16：n-split 比 default(M-split) +31%，但 t32/t64 收敛 → 边际，暂不动（动 use_n_split 风险大）。
- **bf16 单核利用率仅 ~48%（峰值 185.4）**，远低于 i8 的 96% → 这是**微内核/峰值标定**问题（非调度），需单独排查（perf 看 bfmmla 发射率 / 确认 185.4 峰值口径），不在本次调度迁移范围。

## 5. 结论
i8 的调度优化迁移到 bf16：**窄 N 修复迁移成功（+36–85%，已落地 n_tiles>=4 门控）**；t79/GEMV-clamp 不适用（结构不同/已处理）；剩 bf16 单核微内核利用率偏低是独立的后续项。

## 6. ✅ 华为复测确认（n_tiles>=4 门控生效）
重跑 `huawei_bf16_probe.sh`（commit 77101db 后）：`2048×16384×24` 的 default 已从 2D 切到 M-split：
t32 632→**855(+35%)**、t48 752→**1203(+60%)**、t64 795→**1456(+83%)**；default 现 == 最优(m/n)。
`2048×4096×64`(n_tiles=4) 仍走 2D=最优（无回退）；大形状/GEMV 不变。**bf16 窄 N 修复闭环。**
（残留：`256³ @t16` n-split 比 default(M-split) +33%，但 t32/t64 收敛——边际、不动；bf16 单核 ~48% 微内核问题独立待查。）

## 7. ✅ bf16 单核 "48%" 之谜解决 — 是峰值标定错，bf16 已接近最优
跑 `tests/experimental/huawei_bf16_roofline`（华为，taskset 单核，2.9GHz）：
```
bf16_acc8/16/24/30:  ~92.7 GFLOPS(2op)   bfmmla/cycle ≈ 0.50  (跨累加器全平)
```
**华为 bfmmla 实测只发射 0.5/cycle**（throughput-capped，加累加器无用）→ **bf16 真实单核上限 = ~92.8
GFLOPS(2op)**，不是 185.4。185.4 是把 cpufb 92.72 误 ×2（和 i8 当初 370.82→误标 741.6 同样的错）。

重算利用率：fine_sweep 的 bf16 单核最好 ~48%×(185.4/92.8) = **~96% of 真实上限**；均值 36%→**72%**。
→ **bf16 微内核已接近 bfmmla 天花板，无需优化内核**。i8 微内核同理（96% of 370.9）。两者都 compute-bound
在各自的 mmla 发射率上（i8 1/cycle、bf16 0.5/cycle），微内核不是杠杆。

**动作**：把 bf16 peak 口径从 185.4 改成 **92.8**（`huawei_fine_sweep.sh` DT=bf16 默认已改；华为文档同步）。
**结论**：bf16 这轮 = 窄 N 调度修复（+35–83%，已落地）+ 峰值口径修正（bf16 实已 ~96% 单核最优）。无微内核工作项。
