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
