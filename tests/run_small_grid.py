#!/usr/bin/env python3
"""Small-shape sweep: i8gemm vs ACL vs KleidiAI across cache regimes."""
import os, subprocess, sys
from pathlib import Path

T = Path(__file__).resolve().parent
B = T / "build"
BINS = {
    "i8gemm": B / "bench_dispatch_i8gemm_sve",
    "acl":    B / "bench_acl_dispatch",
    "kai":    B / "bench_kleidiai_dispatch",
}
# i8 per-core peak (GOPS) / bf16 per-core peak (GFLOPS), harness 2-ops/MAC units
PEAK = {"i8": 661.22, "bf16": 330.96}

# square-ish sweep (working set i8 ~ 6*d^2 bytes): d=64 ->24KB(L1), 128->98KB,
# 256->393KB(L2), 512->1.5MB(>L2); plus a few skewed small-M / small-K cases.
SQUARE = [
    (16,16,16),(32,32,32),(48,48,48),(64,64,64),(96,96,96),
    (128,128,128),(192,192,192),(256,256,256),(384,384,384),(512,512,512),
    (8,256,256),(16,512,512),(64,64,512),(256,64,64),(16,1024,64),
]
# only-K-small: M,N moderate-large, K tiny -> low arithmetic intensity, B/A streamed
KSMALL = [(256,16,256),(256,32,256),(256,64,256),(256,128,256),
          (512,16,512),(512,32,512),(512,64,512),(512,128,512)]
# only-M-small: tall-skinny LHS (GEMV-like) -> M in {1,2,4,8,16,32}
MSMALL = [(1,512,512),(2,512,512),(4,512,512),(8,512,512),(16,512,512),(32,512,512),
          (4,2048,2048),(8,2048,2048)]
# only-N-small: narrow output
NSMALL = [(512,512,16),(512,512,32),(512,512,64),(512,512,128),
          (256,2048,16),(256,2048,32),(256,2048,64),(256,2048,128)]
SETS = {"square": SQUARE, "ksmall": KSMALL, "msmall": MSMALL, "nsmall": NSMALL}
THREADS = [1, 4]

def reps_for(m,k,n):
    ops = 2*m*k*n
    if ops < 2_000_000: return (2000,200,5)
    if ops < 32_000_000: return (300,30,5)
    if ops < 256_000_000: return (60,10,5)
    return (15,4,4)

def run(bin_, lib, dtype, m,k,n,t):
    reps,wu,runs = reps_for(m,k,n)
    if lib == "i8gemm":
        cmd = [str(bin_), "sve", dtype, str(m),str(k),str(n), str(reps),str(wu),str(runs), str(t)]
        idx = 10
    else:
        cmd = [str(bin_), dtype, str(m),str(k),str(n), str(reps),str(wu),str(runs), str(t)]
        idx = 9
    env = os.environ.copy()
    env["GOMP_CPU_AFFINITY"] = "0-7"; env["OMP_PLACES"]="cores"; env["OMP_PROC_BIND"]="close"
    out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    line = out.stdout.strip().splitlines()
    if not line: return None, ""
    cols = line[-1].split(",")
    try: perf = float(cols[idx])
    except: return None, line[-1]
    cache = cols[2] if lib=="i8gemm" else cols[2]
    return perf, cache

def main():
    args = sys.argv[1:]
    setname = "square"
    if args and args[0] in SETS:
        setname = args[0]; args = args[1:]
    shapes = SETS[setname]
    dtypes = args or ["i8","bf16"]
    for dtype in dtypes:
        print(f"\n==== set={setname} dtype={dtype}  (peak/core={PEAK[dtype]:.0f}; perf, %core-peak) ====")
        print(f"{'M':>4} {'K':>5} {'N':>5} {'thr':>3} {'cache':>6} | "
              f"{'i8gemm':>16} {'ACL':>16} {'KleidiAI':>16} | winner")
        for (m,k,n) in shapes:
            for t in THREADS:
                res = {}
                cache = "?"
                for lib in ("i8gemm","acl","kai"):
                    p,c = run(BINS[lib], lib, dtype, m,k,n,t)
                    res[lib]=p
                    if c: cache=c
                def fmt(p):
                    if p is None: return f"{'-':>16}"
                    pct = 100.0*p/(PEAK[dtype]*t)
                    return f"{p:8.1f}({pct:4.0f}%)"
                vals = {k_:v for k_,v in res.items() if v is not None}
                win = max(vals, key=vals.get) if vals else "-"
                print(f"{m:>4} {k:>5} {n:>5} {t:>3} {cache:>6} | "
                      f"{fmt(res['i8gemm'])} {fmt(res['acl'])} {fmt(res['kai'])} | {win}")

if __name__ == "__main__":
    main()
