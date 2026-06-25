#!/usr/bin/env python3
"""Focused SVE-i8 vs ACL-i8 gap sweep. Reports ratio = sve/acl per (shape,threads).
Highlights shapes where ACL wins (ratio < 0.97)."""
import os, subprocess, sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
BUILD = TESTS / "build"
SVE = BUILD / "bench_dispatch_i8gemm_sve"
ACL = BUILD / "bench_acl_dispatch"

# representative shape set: cubes, skewed, large
SHAPES = [
    # tiny / small cubes
    (16,128,128),(32,256,256),(64,256,256),(64,512,512),(96,512,512),
    (128,128,128),(128,256,256),(128,512,512),
    # medium cubes (historic gap zone)
    (160,512,512),(192,512,512),(224,512,512),(256,256,256),(256,512,512),
    (256,1024,1024),(384,512,512),(512,512,512),(512,1024,1024),
    # small-M skew (large N/K, few rows)  <-- M-split lib target
    (8,512,512),(8,2048,2048),(16,2048,2048),(32,4096,4096),(64,4096,1024),
    (8,4096,4096),(16,4096,8192),(48,2048,2048),
    # large-N skew
    (64,512,4096),(128,512,8192),(256,1024,8192),
    # big
    (512,2048,2048),(1024,2048,2048),(2048,2048,2048),(2048,4096,8192),
]
THREADS = [1,2,4,8]

def reps_for(m,k,n):
    ops = 2*m*k*n
    if ops < 32_000_000: return 50
    if ops < 256_000_000: return 20
    if ops < 2_000_000_000: return 8
    return 3

def runs_for(m,k,n):
    return 4

def get_perf(exe, args, perf_idx):
    env = os.environ.copy()
    env["GOMP_CPU_AFFINITY"]="0-7"; env["OMP_PLACES"]="cores"; env["OMP_PROC_BIND"]="close"
    p = subprocess.run([str(exe)]+args, text=True, capture_output=True, env=env)
    out = p.stdout.strip()
    if not out: return None
    cols = out.split(",")
    try: return float(cols[perf_idx])
    except: return None

def main():
    out = open(TESTS/".."/"results"/"m8"/"gap_sweep_sve_vs_acl.csv","w")
    out.write("M,K,N,threads,sve,acl,ratio\n")
    losses=[]
    for (m,k,n) in SHAPES:
        reps=str(reps_for(m,k,n)); runs=str(runs_for(m,k,n))
        for t in THREADS:
            sve = get_perf(SVE, ["i8gemm_sve","i8",str(m),str(k),str(n),reps,"1",runs,str(t)], 10)
            acl = get_perf(ACL, ["i8",str(m),str(k),str(n),reps,"1",runs,str(t)], 9)
            if sve is None or acl is None or acl==0:
                print(f"SKIP {m}x{k}x{n} t{t} sve={sve} acl={acl}", file=sys.stderr); continue
            r = sve/acl
            out.write(f"{m},{k},{n},{t},{sve:.1f},{acl:.1f},{r:.3f}\n"); out.flush()
            tag = "  <-- ACL WINS" if r < 0.97 else ("  (close)" if r<1.0 else "")
            line=f"{m:5d}x{k:5d}x{n:5d} t{t}: sve={sve:8.1f} acl={acl:8.1f} ratio={r:.3f}{tag}"
            print(line)
            if r < 0.97: losses.append((r,m,k,n,t,sve,acl))
    print("\n==== SHAPES WHERE ACL WINS (ratio<0.97), worst first ====")
    for r,m,k,n,t,sve,acl in sorted(losses):
        print(f"  {m}x{k}x{n} t{t}: ratio={r:.3f} (sve {sve:.0f} vs acl {acl:.0f})")
    out.close()

if __name__=="__main__":
    main()
