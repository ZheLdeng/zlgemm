#!/usr/bin/env python3
"""Run M/K/N grid benchmarks for i8gemm and available third-party adapters."""

from __future__ import annotations

import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
I8GEMM = ROOT / "i8gemm"
TESTS = I8GEMM / "tests"
LIB = I8GEMM / "lib"
BUILD = TESTS / "build"
RESULTS = I8GEMM / "results" / "m8"
ACL_BUILD = ROOT / "ComputeLibrary" / "build-codex"
KLEIDIAI = ROOT / "kleidiai"

M_VALUES = [16, 32, 64, 128, 256, 512, 1024, 2048]
K_VALUES = [128, 256, 512, 1024, 2048, 4096]
N_VALUES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
THREADS = [1, 2, 4, 8]


def run(cmd: list[str], cwd: Path = ROOT, env: dict[str, str] | None = None) -> str:
    print("+ " + " ".join(shlex.quote(x) for x in cmd), file=sys.stderr)
    p = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout)
        sys.stderr.write(p.stderr)
        raise SystemExit(p.returncode)
    if p.stderr:
        sys.stderr.write(p.stderr)
    return p.stdout


def reps_for(m: int, k: int, n: int) -> int:
    override = os.environ.get("REPS_OVERRIDE")
    if override:
        return int(override)
    ops = 2 * m * k * n
    if ops < 32_000_000:
        return 50
    if ops < 256_000_000:
        return 20
    if ops < 2_000_000_000:
        return 8
    return 3


def build_i8gemm() -> list[tuple[str, Path]]:
    BUILD.mkdir(parents=True, exist_ok=True)
    common = [
        "cc",
        "-O3",
        "-Wall",
        "-fopenmp",
        "-mcpu=native",
        f"-I{LIB}",
    ]
    sve_src = [
        TESTS / "bench_dispatch_types.c",
        LIB / "bf16gemm_sve.c",
        LIB / "bf16gemm_sve.S",
        LIB / "i8gemm_sve.c",
        LIB / "i8gemm_m16n4.c",
        LIB / "i8gemm_m16n4.S",
        LIB / "i8gemm_sve.S",
        LIB / "i8gemm_hybrid.S",
        LIB / "i8gemm_pack_a_neon.S",
    ]
    neon_src = [
        TESTS / "bench_dispatch_types.c",
        LIB / "bf16gemm_mt.c",
        LIB / "bf16gemm_k.S",
        LIB / "bf16gemm_k_bias.S",
        LIB / "i8gemm_mt.c",
        LIB / "i8gemm_m16n4.c",
        LIB / "i8gemm_m16n4.S",
        LIB / "i8gemm_k.S",
        LIB / "i8gemm_k_bias.S",
        LIB / "i8gemm_pack_a_neon.S",
    ]
    sve = BUILD / "bench_dispatch_i8gemm_sve"
    neon = BUILD / "bench_dispatch_i8gemm_neon"
    run(common + ["-DBENCH_SVE", "-o", str(sve)] + [str(x) for x in sve_src] + ["-lm"])
    run(common + ["-o", str(neon)] + [str(x) for x in neon_src] + ["-lm"])
    return [("i8gemm_sve", sve), ("i8gemm_neon", neon), ("i8gemm_neon_m16n4", neon)]


def build_acl() -> Path | None:
    lib_acl = ACL_BUILD / "libarm_compute.so"
    if not lib_acl.exists():
        print("ACL adapter skipped: libarm_compute.so is not built yet", file=sys.stderr)
        return None
    exe = BUILD / "bench_acl_dispatch"
    cmd = [
        "c++",
        "-O3",
        "-Wall",
        "-std=c++17",
        "-fopenmp",
        "-mcpu=native",
        f"-I{ROOT / 'ComputeLibrary'}",
        f"-I{ROOT / 'ComputeLibrary' / 'include'}",
        f"-I{ROOT / 'ComputeLibrary' / 'src' / 'cpu' / 'kernels' / 'assembly'}",
        "-o",
        str(exe),
        str(TESTS / "bench_acl_dispatch.cpp"),
        f"-L{ACL_BUILD}",
        "-larm_compute",
        "-Wl,-rpath," + str(ACL_BUILD),
        "-lpthread",
        "-lm",
    ]
    run(cmd)
    return exe


def build_kleidiai() -> Path | None:
    if not KLEIDIAI.exists():
        print("KleidiAI adapter skipped: kleidiai checkout is missing", file=sys.stderr)
        return None
    exe = BUILD / "bench_kleidiai_dispatch"
    srcs = [
        TESTS / "bench_kleidiai_dispatch.c",
        KLEIDIAI / "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla.c",
        KLEIDIAI / "kai/ukernels/matmul/pack/kai_lhs_quant_pack_bf16p8x4_f32_neon.c",
        KLEIDIAI / "kai/ukernels/matmul/pack/kai_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon.c",
        KLEIDIAI / "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.c",
        KLEIDIAI / "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c",
        KLEIDIAI / "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi8cxp_qsi8cx_neon.c",
    ]
    cmd = [
        "cc",
        "-O3",
        "-Wall",
        "-fopenmp",
        "-mcpu=native",
        f"-I{KLEIDIAI}",
        "-o",
        str(exe),
    ] + [str(x) for x in srcs] + ["-lm"]
    run(cmd)
    return exe


def parse_i8gemm_line(line: str, status: str = "ok", note: str = "") -> dict[str, str]:
    cols = line.strip().split(",")
    if len(cols) < 11:
        raise ValueError(line)
    return {
        "impl": cols[0],
        "dtype": cols[1],
        "cache": cols[2],
        "M": cols[3],
        "K": cols[4],
        "N": cols[5],
        "threads": cols[6],
        "KiB": cols[8],
        "reps": cols[9],
        "perf": cols[10],
        "status": status,
        "note": note,
    }


def parse_acl_line(line: str) -> dict[str, str]:
    cols = line.strip().split(",", 11)
    if len(cols) != 12:
        raise ValueError(line)
    return {
        "impl": cols[0],
        "dtype": cols[1],
        "cache": cols[2],
        "M": cols[3],
        "K": cols[4],
        "N": cols[5],
        "threads": cols[6],
        "KiB": cols[7],
        "reps": cols[8],
        "perf": cols[9],
        "status": cols[10],
        "note": cols[11],
    }


def main() -> None:
    out = Path(os.environ.get("OUT", RESULTS / "three_lib_grid_8c.csv"))
    quick = os.environ.get("QUICK", "0") != "0"
    coreset = os.environ.get("CORESET", "0-7")
    warmup = os.environ.get("WARMUP", "1")
    runs = os.environ.get("RUNS", "2")
    RESULTS.mkdir(parents=True, exist_ok=True)

    benches = [("i8gemm", name, exe) for name, exe in build_i8gemm()]
    acl_exe = build_acl()
    if acl_exe:
        benches.append(("acl", "acl_auto", acl_exe))
    kleidiai_exe = build_kleidiai()
    if kleidiai_exe:
        benches.append(("kleidiai", "kleidiai_fixed", kleidiai_exe))

    shapes = [(m, k, n) for k in K_VALUES for m in M_VALUES for n in N_VALUES]
    if quick:
        shapes = [
            (16, 128, 128),
            (64, 512, 512),
            (512, 1024, 1024),
            (2048, 4096, 8192),
        ]

    with out.open("w", newline="") as f:
        fieldnames = [
            "lib",
            "impl",
            "dtype",
            "cache",
            "M",
            "K",
            "N",
            "threads",
            "KiB",
            "reps",
            "perf",
            "status",
            "note",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m, k, n in shapes:
            reps = str(reps_for(m, k, n))
            for t in THREADS:
                env = os.environ.copy()
                env["GOMP_CPU_AFFINITY"] = coreset
                env["OMP_PLACES"] = "cores"
                env["OMP_PROC_BIND"] = "close"
                for lib, impl, exe in benches:
                    dtypes = ["bf16", "i8"]
                    if impl == "i8gemm_neon_m16n4":
                        dtypes = ["i8"]
                    for dtype in dtypes:
                        if lib == "i8gemm":
                            cmd = [
                                str(exe),
                                impl,
                                dtype,
                                str(m),
                                str(k),
                                str(n),
                                reps,
                                warmup,
                                runs,
                                str(t),
                            ]
                            row = parse_i8gemm_line(run(cmd, env=env).strip())
                        else:
                            cmd = [
                                str(exe),
                                dtype,
                                str(m),
                                str(k),
                                str(n),
                                reps,
                                warmup,
                                runs,
                                str(t),
                            ]
                            row = parse_acl_line(run(cmd, env=env).strip())
                        row["lib"] = lib
                        writer.writerow(row)
                        f.flush()
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
