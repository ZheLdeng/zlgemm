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
