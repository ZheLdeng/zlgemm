#!/usr/bin/env bash
set -euo pipefail

# Build and run M8 BF16 performance attribution variants.
#
# Defaults:
#   K values: 128 256 1024 2048
#   M/N:      16 32 64 128 256 512 1024
#   reps:     100
#
# Output CSV columns:
#   variant,mode,cache,M,K,N,KiB,reps,GFLOPS,pct_of_330
#
# Useful overrides:
#   REPS=1 ./run_m8_parts.sh
#   K_VALUES="128 256" PART_VARIANTS="full nostore noload" ./run_m8_parts.sh
#   STORE_IMPLS="sve neon" STORE_MODES="f32 bf16 bias" ./run_m8_parts.sh
#   OUT=/tmp/m8.csv ./run_m8_parts.sh

CC=${CC:-cc}
ARCH_FLAGS=${ARCH_FLAGS:-"-march=armv8.6-a+sve+bf16"}
OPT_FLAGS=${OPT_FLAGS:-"-O3"}
SVE_SRC=${SVE_SRC:-"bf16gemm_sve_m8_nld.S"}
NEON_SRC=${NEON_SRC:-"bf16gemm_neon_m8_nld.S"}
BENCH=${BENCH:-"bench_m8_parts.c"}
OUT=${OUT:-"m8_parts_results.csv"}
REPS=${REPS:-100}
WARMUP=${WARMUP:-50}
RUNS=${RUNS:-7}
K_VALUES=${K_VALUES:-"128 256 1024 2048"}
M_VALUES=${M_VALUES:-"16 32 64 128 256 512 1024"}
N_VALUES=${N_VALUES:-"16 32 64 128 256 512 1024"}
PART_VARIANTS=${PART_VARIANTS:-"full prepacked nozero nostore noload nozero_nostore noload_nostore compute_only"}
STORE_IMPLS=${STORE_IMPLS:-"sve neon"}
STORE_MODES=${STORE_MODES:-"f32 bf16 bias"}
RUN_PARTS=${RUN_PARTS:-1}
RUN_STORES=${RUN_STORES:-1}

WORKDIR=${WORKDIR:-"/tmp/m8_parts_${USER}_$$"}
mkdir -p "$WORKDIR"
trap 'rm -rf "$WORKDIR"' EXIT

generate_variant() {
    local variant=$1
    local out_s=$2
    python3 - "$SVE_SRC" "$variant" "$out_s" <<'PY'
import sys

src, variant, out_s = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(src, "r", encoding="utf-8").read().splitlines()

flags = {
    "full": set(),
    "prepacked": {"no_reorder"},
    "nozero": {"no_zero"},
    "nostore": {"no_store"},
    "noload": {"no_ab_load", "init_ab"},
    "nozero_nostore": {"no_zero", "no_store"},
    "noload_nostore": {"no_ab_load", "no_store", "init_ab"},
    "compute_only": {"no_reorder", "no_zero", "no_store", "no_ab_load", "init_ab"},
}[variant]

init_ab = [
    "\tmov z0.h, #1",
    "\tmov z1.h, #1",
    "\tmov z2.h, #1",
    "\tmov z3.h, #1",
    "\tmov z4.h, #1",
    "\tmov z5.h, #1",
    "\tmov z6.h, #1",
    "\tmov z7.h, #1",
    "\tmov z8.h, #1",
    "\tmov z9.h, #1",
    "\tmov z10.h, #1",
    "\tmov z11.h, #1",
    "\tmov z12.h, #1",
    "\tmov z13.h, #1",
    "\tmov z14.h, #1",
    "\tmov z15.h, #1",
]

out = []
for line in lines:
    stripped = line.strip()
    if "no_reorder" in flags and stripped == "REORDER_A_M8":
        out.append("//" + line)
        continue
    if "no_zero" in flags and stripped == "ZERO_ACC":
        out.append("//" + line)
        continue
    if "no_store" in flags and stripped in ("BUILD_F32_OFFSETS_AND_BASES", "STORE_8XNTILE_F32"):
        out.append("//" + line)
        continue
    if "no_ab_load" in flags and (line.startswith("\tld1h") or line.startswith("\tld1rqh")):
        out.append("//" + line)
        continue
    out.append(line)
    if "init_ab" in flags and stripped == "ptrue p1.s":
        out.extend(init_ab)

open(out_s, "w", encoding="utf-8").write("\n".join(out) + "\n")
PY
}

build_variant() {
    local variant=$1
    local src_s="$WORKDIR/${variant}.S"
    local obj="$WORKDIR/${variant}.o"
    local bin="$WORKDIR/bench_${variant}"
    generate_variant "$variant" "$src_s"
    "$CC" $OPT_FLAGS $ARCH_FLAGS -c "$src_s" -o "$obj"
    "$CC" $OPT_FLAGS $ARCH_FLAGS "$BENCH" "$obj" -o "$bin"
}

build_source() {
    local label=$1
    local src=$2
    local obj="$WORKDIR/${label}.o"
    local bin="$WORKDIR/bench_${label}"
    "$CC" $OPT_FLAGS $ARCH_FLAGS -c "$src" -o "$obj"
    "$CC" $OPT_FLAGS $ARCH_FLAGS "$BENCH" "$obj" -o "$bin"
}

if [[ "$RUN_PARTS" != "0" ]]; then
    for variant in $PART_VARIANTS; do
        build_variant "$variant"
    done
fi

if [[ "$RUN_STORES" != "0" ]]; then
    for impl in $STORE_IMPLS; do
        case "$impl" in
            sve)  build_source "store_sve" "$SVE_SRC" ;;
            neon) build_source "store_neon" "$NEON_SRC" ;;
            *) echo "unknown impl: $impl" >&2; exit 2 ;;
        esac
    done
fi

echo "variant,mode,cache,M,K,N,KiB,reps,GFLOPS,pct_of_330" > "$OUT"
for K in $K_VALUES; do
    for M in $M_VALUES; do
        for N in $N_VALUES; do
            if [[ "$RUN_PARTS" != "0" ]]; then
                for variant in $PART_VARIANTS; do
                    "$WORKDIR/bench_${variant}" "sve_${variant}" "$M" "$K" "$N" "$REPS" "$WARMUP" "$RUNS" f32 >> "$OUT"
                done
            fi
            if [[ "$RUN_STORES" != "0" ]]; then
                for impl in $STORE_IMPLS; do
                    for mode in $STORE_MODES; do
                        "$WORKDIR/bench_store_${impl}" "${impl}_${mode}" "$M" "$K" "$N" "$REPS" "$WARMUP" "$RUNS" "$mode" >> "$OUT"
                    done
                done
            fi
        done
    done
done

echo "wrote $OUT"
