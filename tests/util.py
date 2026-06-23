import subprocess, time
PK={'i8':661.22,'bf16':330.96}
shapes=[(64,512,512),(128,512,1024),(256,512,1024),(512,1024,1024),(2048,4096,8192)]
def run(dt,M,K,N,t):
    reps="6 2 4" if M*K*N>1e9 else "30 5 5"
    out=subprocess.run(["./build/bench_util","sve",dt,str(M),str(K),str(N)]+reps.split()+[str(t)],
                       capture_output=True,text=True).stdout.strip().splitlines()
    return float(out[-1].split(',')[10]) if out else 0.0
for dt in ('i8','bf16'):
    print("\n=== %s (perf, %% of threads*per-core-peak %.0f) ===" % (dt,PK[dt]))
    print("%5s%6s%6s | "%("M","K","N") + " ".join("%13s"%("%dt"%t) for t in (1,2,4,8)))
    for (M,K,N) in shapes:
        cells=[]
        for t in (1,2,4,8):
            time.sleep(0.3); g=run(dt,M,K,N,t); pct=100*g/(PK[dt]*t)
            cells.append("%6.0f(%2.0f%%)"%(g,pct))
        print("%5d%6d%6d | "%(M,K,N)+" ".join("%13s"%c for c in cells))
