#!/usr/bin/env bash
set -euo pipefail

# Build and run M8 BF16 performance attribution variants.
#
# Defaults:
#   CASE_MODE: shape
#   Shape-pruned cases are M,K,N triples derived from shape.csv buckets.
#   Use CASE_MODE=grid to run the full K x M x N Cartesian sweep.
#   Grid K values: 128 256 512 1024 2048 4096
#   Grid M values: 16 32 64 128 256 512 1024 2048
#   Grid N values: 16 32 64 128 256 512 1024 2048 4096 8192
#   reps:     100
#   threads:  1
#
# Output workbook:
#   m8_results.xlsx with sheets: parts, tail
#
# Parts sheet columns:
#   variant,mode,cache,M,K,N,threads,KiB,reps,GFLOPS,pct_of_330,pct_of_330xthreads
#
# Tail sheet columns:
#   impl,mode,cache,base_M,tail_M,K,N,threads,KiB,reps,base_GFLOPS,tail_GFLOPS,
#   tail_vs_base_pct,tail_drop_pct,pct_of_330,pct_of_330xthreads
#
# Useful overrides:
#   REPS=1 ./run_m8_parts.sh
#   THREADS="1 2 4 8" ./run_m8_parts.sh
#   THREADS=auto c=64-79 ./run_m8_parts.sh
#   CORES=64,67,68 THREADS="1 2 4" ./run_m8_parts.sh
#   NCORES=16 THREADS=auto ./run_m8_parts.sh
#   K_VALUES="128 256" PART_VARIANTS="full nostore noload" ./run_m8_parts.sh
#   STORE_IMPLS="sve neon" STORE_MODES="f32 bf16 bias" ./run_m8_parts.sh
#   RUN_STRIDE=1 STRIDE_FACTORS="1 2" ./run_m8_parts.sh
#   RUN_BATCH=1 BATCH_COUNTS="1 4 16" ./run_m8_parts.sh
#   RUN_DISPATCH=1 DISPATCH_IMPLS="sve neon" DISPATCH_DTYPES="bf16 i8" ./run_m8_parts.sh
#   RUN_TAILS=1 TAIL_BASE_M_VALUES=16 TAIL_DELTAS="1 2 4" ./run_m8_parts.sh
#   RESULTS_XLSX=/tmp/m8.xlsx ./run_m8_parts.sh
#   KEEP_CSV=1 OUT=/tmp/m8_parts.csv TAIL_OUT=/tmp/m8_tail.csv ./run_m8_parts.sh
#   PROGRESS=0 ./run_m8_parts.sh
#   CASE_MODE=grid ./run_m8_parts.sh
#   CASE_MODE=lite ./run_m8_parts.sh
#   CASES="64,512,4096 2048,4096,1024" ./run_m8_parts.sh
#   PRUNE_BIG_CASE_THREADS=0 ./run_m8_parts.sh
#   PRUNE_TOTAL_BYTES=$((3*1024*1024)) PRUNE_PER_THREAD_BYTES=$((4*1024*1024)) ./run_m8_parts.sh

CC=${CC:-cc}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
LIB_DIR=${LIB_DIR:-"$ROOT_DIR/lib"}
RESULTS_DIR=${RESULTS_DIR:-"$ROOT_DIR/results/m8"}
mkdir -p "$RESULTS_DIR"
ARCH_FLAGS=${ARCH_FLAGS:-"-march=armv8.6-a+sve+bf16"}
OPT_FLAGS=${OPT_FLAGS:-"-O3"}
OMP_FLAGS=${OMP_FLAGS:-"-fopenmp"}
INCLUDE_FLAGS=${INCLUDE_FLAGS:-"-I$LIB_DIR -I$SCRIPT_DIR"}
SVE_SRC=${SVE_SRC:-"$SCRIPT_DIR/bf16gemm_sve_m8_nld.S"}
NEON_SRC=${NEON_SRC:-"$SCRIPT_DIR/bf16gemm_neon_m8_nld.S"}
BENCH=${BENCH:-"$SCRIPT_DIR/bench_m8_parts.c"}
TAIL_BENCH=${TAIL_BENCH:-"$SCRIPT_DIR/bench_full_tail.c"}
DISPATCH_BENCH=${DISPATCH_BENCH:-"$SCRIPT_DIR/bench_dispatch_types.c"}
OUT_WAS_SET=${OUT+x}
TAIL_OUT_WAS_SET=${TAIL_OUT+x}
STRIDE_OUT_WAS_SET=${STRIDE_OUT+x}
BATCH_OUT_WAS_SET=${BATCH_OUT+x}
DISPATCH_OUT_WAS_SET=${DISPATCH_OUT+x}
OUT=${OUT:-"$RESULTS_DIR/m8_parts_results.csv"}
TAIL_OUT=${TAIL_OUT:-"$RESULTS_DIR/m8_tail_results.csv"}
STRIDE_OUT=${STRIDE_OUT:-"$RESULTS_DIR/m8_stride_results.csv"}
BATCH_OUT=${BATCH_OUT:-"$RESULTS_DIR/m8_batch_results.csv"}
DISPATCH_OUT=${DISPATCH_OUT:-"$RESULTS_DIR/m8_dispatch_results.csv"}
RESULTS_XLSX=${RESULTS_XLSX:-"$RESULTS_DIR/m8_results.xlsx"}
WRITE_XLSX=${WRITE_XLSX:-1}
KEEP_CSV=${KEEP_CSV:-0}
K_VALUES=${K_VALUES:-"128 256 512 1024 2048 4096"}
M_VALUES=${M_VALUES:-"16 32 64 128 256 512 1024 2048"}
N_VALUES=${N_VALUES:-"16 32 64 128 256 512 1024 2048 4096 8192"}
CASE_MODE=${CASE_MODE:-shape}
# Shape case format is M,K,N. N=24 is represented by N=32 here because the
# standalone M8 parts kernels operate on a full SVE/NEON N tile.
DEFAULT_CASES="
16,512,4096 16,4096,1024
32,512,4096 32,4096,1024
64,512,4096 64,4096,1024
128,512,4096 128,4096,1024
256,512,4096 256,4096,1024
512,512,4096 512,4096,1024
1024,512,4096 1024,4096,1024
2048,512,4096 2048,4096,1024
64,128,512 64,256,512 64,512,128 64,512,256
64,512,640 64,640,512
2048,128,512 2048,1024,8192 2048,2048,4096
2048,4096,64 2048,4096,256 2048,4096,512
2048,4096,1536 2048,4096,2048
2048,16384,32
16,4,4096 16,4096,32320
"
LITE_CASES=${LITE_CASES:-"
16,512,128
16,512,4096
64,512,512
64,512,4096
64,4096,64
64,4096,1024
128,4096,1024
512,512,4096
512,4096,1024
2048,512,4096
2048,1024,8192
2048,4096,64
"}
if [[ "$CASE_MODE" == "lite" ]]; then
    REPS=${REPS:-20}
    WARMUP=${WARMUP:-5}
    RUNS=${RUNS:-3}
    THREADS=${THREADS:-"1 2 4 8"}
    CASES=${CASES:-"$LITE_CASES"}
    RUN_TAILS=${RUN_TAILS:-1}
    TAIL_CASES=${TAIL_CASES:-"512,4096,1024 64,512,4096 2048,4096,64"}
else
    REPS=${REPS:-100}
    WARMUP=${WARMUP:-50}
    RUNS=${RUNS:-7}
    THREADS=${THREADS:-"1"}
    CASES=${CASES:-"$DEFAULT_CASES"}
fi
TAIL_CASES=${TAIL_CASES:-"$CASES"}
DEFAULT_THREADS=${DEFAULT_THREADS:-"1 2 4 8 10 16 20 32 40 64 80"}
NPROC=$(nproc 2>/dev/null || echo 64)
LAST=$((NPROC - 1))
c=${c:-}
CORES=${CORES:-${c}}
CORESET=${CORESET:-${CORES}}
NCORES=${NCORES:-}
SKIP_OVERSUB=${SKIP_OVERSUB:-1}
OMP_PLACES=${OMP_PLACES:-cores}
OMP_PROC_BIND=${OMP_PROC_BIND:-close}
PART_VARIANTS=${PART_VARIANTS:-"full prepacked nozero nostore noload nozero_nostore noload_nostore"}
STORE_IMPLS=${STORE_IMPLS:-"sve neon"}
STORE_MODES=${STORE_MODES:-"f32 bf16 bias"}
STRIDE_IMPLS=${STRIDE_IMPLS:-"sve neon"}
STRIDE_MODES=${STRIDE_MODES:-"f32"}
STRIDE_FACTORS=${STRIDE_FACTORS:-"1 2"}
BATCH_IMPLS=${BATCH_IMPLS:-"sve neon"}
BATCH_MODES=${BATCH_MODES:-"f32"}
BATCH_COUNTS=${BATCH_COUNTS:-"1 4 16"}
DISPATCH_IMPLS=${DISPATCH_IMPLS:-"sve neon"}
DISPATCH_DTYPES=${DISPATCH_DTYPES:-"bf16 i8"}
DISPATCH_BATCH_COUNTS=${DISPATCH_BATCH_COUNTS:-"1"}
BASELINE_IMPLS=${BASELINE_IMPLS:-"sve neon"}
TAIL_IMPLS=${TAIL_IMPLS:-"sve neon"}
TAIL_BASE_M_VALUES=${TAIL_BASE_M_VALUES:-"$M_VALUES"}
TAIL_DELTAS=${TAIL_DELTAS:-"1 2 4"}
TAIL_N_ALIGN=${TAIL_N_ALIGN:-0}
RUN_PARTS=${RUN_PARTS:-1}
RUN_STORES=${RUN_STORES:-1}
RUN_BASELINES=${RUN_BASELINES:-1}
RUN_TAILS=${RUN_TAILS:-1}
RUN_STRIDE=${RUN_STRIDE:-0}
RUN_BATCH=${RUN_BATCH:-0}
RUN_DISPATCH=${RUN_DISPATCH:-0}
PROGRESS=${PROGRESS:-1}
PRUNE_BIG_CASE_THREADS=${PRUNE_BIG_CASE_THREADS:-1}
PRUNE_TOTAL_BYTES=${PRUNE_TOTAL_BYTES:-$((3 * 1024 * 1024))}
PRUNE_PER_THREAD_BYTES=${PRUNE_PER_THREAD_BYTES:-$((4 * 1024 * 1024))}

WORKDIR=${WORKDIR:-"/tmp/m8_parts_${USER}_$$"}
mkdir -p "$WORKDIR"
trap 'rm -rf "$WORKDIR"' EXIT

if [[ "$KEEP_CSV" == "0" ]]; then
    if [[ -z "${OUT_WAS_SET:-}" ]]; then
        OUT="$WORKDIR/m8_parts_results.csv"
    fi
    if [[ -z "${TAIL_OUT_WAS_SET:-}" ]]; then
        TAIL_OUT="$WORKDIR/m8_tail_results.csv"
    fi
    if [[ -z "${STRIDE_OUT_WAS_SET:-}" ]]; then
        STRIDE_OUT="$WORKDIR/m8_stride_results.csv"
    fi
    if [[ -z "${BATCH_OUT_WAS_SET:-}" ]]; then
        BATCH_OUT="$WORKDIR/m8_batch_results.csv"
    fi
    if [[ -z "${DISPATCH_OUT_WAS_SET:-}" ]]; then
        DISPATCH_OUT="$WORKDIR/m8_dispatch_results.csv"
    fi
fi

count_cpulist() {
    local list=$1
    local total=0
    local part start end
    list=${list// /}
    if [[ -z "$list" ]]; then
        echo "$NPROC"
        return
    fi
    IFS=',' read -ra parts <<< "$list"
    for part in "${parts[@]}"; do
        if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            start=${BASH_REMATCH[1]}
            end=${BASH_REMATCH[2]}
            if (( end >= start )); then
                total=$((total + end - start + 1))
            fi
        elif [[ "$part" =~ ^[0-9]+$ ]]; then
            total=$((total + 1))
        fi
    done
    if (( total <= 0 )); then
        total=1
    fi
    echo "$total"
}

AVAILABLE_CORES=${NCORES:-$(count_cpulist "$CORESET")}
if [[ "$THREADS" == "auto" ]]; then
    THREADS="$DEFAULT_THREADS"
fi

count_words() {
    local n=0
    local x
    for x in "$@"; do
        if [[ -n "$x" ]]; then
            n=$((n + 1))
        fi
    done
    echo "$n"
}

thread_allowed() {
    local t=$1
    [[ "$SKIP_OVERSUB" == "0" ]] || (( t <= AVAILABLE_CORES ))
}

abc_bytes() {
    local M=$1
    local K=$2
    local N=$3
    echo $((2 * M * K + 2 * K * N + 4 * M * N))
}

case_thread_allowed() {
    local M=$1
    local K=$2
    local N=$3
    local T=$4
    local bytes

    thread_allowed "$T" || return 1
    if [[ "$PRUNE_BIG_CASE_THREADS" == "0" ]]; then
        return 0
    fi

    bytes=$(abc_bytes "$M" "$K" "$N")
    if (( bytes <= PRUNE_TOTAL_BYTES )); then
        return 0
    fi
    (( (bytes + T - 1) / T <= PRUNE_PER_THREAD_BYTES ))
}

max_tail_delta() {
    local d
    local max=0
    for d in $TAIL_DELTAS; do
        if (( d > max )); then
            max=$d
        fi
    done
    echo "$max"
}

active_thread_count() {
    local n=0
    local t
    for t in $THREADS; do
        if thread_allowed "$t"; then
            n=$((n + 1))
        fi
    done
    echo "$n"
}

case_count() {
    if [[ "$CASE_MODE" == "grid" ]]; then
        echo $(($(count_words $K_VALUES) * $(count_words $M_VALUES) * $(count_words $N_VALUES)))
    else
        count_words $CASES
    fi
}

tail_case_count() {
    if [[ "$CASE_MODE" == "grid" ]]; then
        echo $(($(count_words $K_VALUES) * $(count_words $TAIL_BASE_M_VALUES) * $(count_words $N_VALUES)))
    else
        count_words $TAIL_CASES
    fi
}

TOTAL_STEPS=0
STEP=0

count_total_steps() {
    local nc nb np ns nsm nti nd nsi nsim nsf nbti nbc ndi ndt ndbc
    nc=$(case_count)
    nb=0
    np=0
    ns=0
    nsm=0
    nti=0
    nd=0
    nsi=0
    nsim=0
    nsf=0
    nbti=0
    nbc=0
    ndi=0
    ndt=0
    ndbc=0
    if [[ "$RUN_BASELINES" != "0" ]]; then
        nb=$(count_words $BASELINE_IMPLS)
    fi
    if [[ "$RUN_PARTS" != "0" ]]; then
        np=$(count_words $PART_VARIANTS)
    fi
    if [[ "$RUN_STORES" != "0" ]]; then
        ns=$(count_words $STORE_IMPLS)
        nsm=$(count_words $STORE_MODES)
    fi
    if [[ "$RUN_TAILS" != "0" ]]; then
        nti=$(count_words $TAIL_IMPLS)
        nd=$(count_words $TAIL_DELTAS)
    fi
    if [[ "$RUN_STRIDE" != "0" ]]; then
        nsi=$(count_words $STRIDE_IMPLS)
        nsim=$(count_words $STRIDE_MODES)
        nsf=$(count_words $STRIDE_FACTORS)
    fi
    if [[ "$RUN_BATCH" != "0" ]]; then
        nbti=$(count_words $BATCH_IMPLS)
        nbc=$(count_words $BATCH_COUNTS)
    fi
    if [[ "$RUN_DISPATCH" != "0" ]]; then
        ndi=$(count_words $DISPATCH_IMPLS)
        ndt=$(count_words $DISPATCH_DTYPES)
        ndbc=$(count_words $DISPATCH_BATCH_COUNTS)
    fi

    local per_case_parts=$((nb + np + ns * nsm))
    local per_case_extra=$((nsi * nsim * nsf + nbti * nbc + ndi * ndt * ndbc))
    local per_tail_impl=$((1 + nd))
    local t M K N case base_M tail_m max_d
    TOTAL_STEPS=0

    if [[ "$CASE_MODE" == "grid" ]]; then
        for K in $K_VALUES; do
            for M in $M_VALUES; do
                for N in $N_VALUES; do
                    for t in $THREADS; do
                        if case_thread_allowed "$M" "$K" "$N" "$t"; then
                            TOTAL_STEPS=$((TOTAL_STEPS + per_case_parts + per_case_extra))
                        fi
                    done
                done
            done
        done
    else
        for case in $CASES; do
            IFS=, read -r M K N <<< "$case"
            for t in $THREADS; do
                if case_thread_allowed "$M" "$K" "$N" "$t"; then
                    TOTAL_STEPS=$((TOTAL_STEPS + per_case_parts + per_case_extra))
                fi
            done
        done
    fi

    if [[ "$RUN_TAILS" != "0" ]]; then
        max_d=$(max_tail_delta)
        if [[ "$CASE_MODE" == "grid" ]]; then
            for K in $K_VALUES; do
                for base_M in $TAIL_BASE_M_VALUES; do
                    for N in $N_VALUES; do
                        tail_m=$((base_M + max_d))
                        for t in $THREADS; do
                            if case_thread_allowed "$tail_m" "$K" "$N" "$t"; then
                                TOTAL_STEPS=$((TOTAL_STEPS + nti * per_tail_impl))
                            fi
                        done
                    done
                done
            done
        else
            for case in $TAIL_CASES; do
                IFS=, read -r base_M K N <<< "$case"
                tail_m=$((base_M + max_d))
                for t in $THREADS; do
                    if case_thread_allowed "$tail_m" "$K" "$N" "$t"; then
                        TOTAL_STEPS=$((TOTAL_STEPS + nti * per_tail_impl))
                    fi
                done
            done
        fi
    fi
}

progress() {
    if [[ "$PROGRESS" == "0" ]]; then
        return
    fi
    STEP=$((STEP + 1))
    printf '[%d/%d] %s\n' "$STEP" "$TOTAL_STEPS" "$*" >&2
}

run_bench() {
    local bin=$1
    shift
    if [[ -n "$CORESET" ]]; then
        OMP_PLACES="$OMP_PLACES" OMP_PROC_BIND="$OMP_PROC_BIND" taskset -c "$CORESET" "$bin" "$@"
    else
        OMP_PLACES="$OMP_PLACES" OMP_PROC_BIND="$OMP_PROC_BIND" "$bin" "$@"
    fi
}

write_xlsx() {
    local xlsx=$1
    shift
    python3 - "$xlsx" "$@" <<'PY'
import csv
import html
import os
import re
import sys
import zipfile

xlsx = sys.argv[1]
args = sys.argv[2:]
if len(args) % 2 != 0:
    raise SystemExit("write_xlsx expects sheet/path pairs")
sheets = []
for i in range(0, len(args), 2):
    name, path = args[i], args[i + 1]
    if path and os.path.exists(path):
        sheets.append((name, path))
if not sheets:
    raise SystemExit("no CSV sheets to write")

num_re = re.compile(r"^-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")

def col_name(idx):
    out = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        out = chr(65 + rem) + out
    return out

def sheet_xml(path):
    rows = []
    with open(path, newline="") as f:
        for r, row in enumerate(csv.reader(f), 1):
            cells = []
            for c, value in enumerate(row):
                ref = f"{col_name(c)}{r}"
                if r > 1 and num_re.match(value):
                    cells.append(f'<c r="{ref}"><v>{value}</v></c>')
                else:
                    text = html.escape(value)
                    cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>')
            rows.append(f'<row r="{r}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData>' + ''.join(rows) + '</sheetData></worksheet>'
    )

content_types = [
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
    '<Default Extension="xml" ContentType="application/xml"/>',
    '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
]
for i, _ in enumerate(sheets, 1):
    content_types.append(
        f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
    )
content_types.append('</Types>')

workbook_sheets = []
rels = [
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
]
for i, (name, _) in enumerate(sheets, 1):
    safe_name = html.escape(name[:31])
    workbook_sheets.append(
        f'<sheet name="{safe_name}" sheetId="{i}" r:id="rId{i}"/>'
    )
    rels.append(
        f'<Relationship Id="rId{i}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{i}.xml"/>'
    )
rels.append('</Relationships>')

workbook = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
    '<sheets>' + ''.join(workbook_sheets) + '</sheets></workbook>'
)
root_rels = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" '
    'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
    'Target="xl/workbook.xml"/></Relationships>'
)

with zipfile.ZipFile(xlsx, "w", compression=zipfile.ZIP_DEFLATED) as z:
    z.writestr("[Content_Types].xml", ''.join(content_types))
    z.writestr("_rels/.rels", root_rels)
    z.writestr("xl/workbook.xml", workbook)
    z.writestr("xl/_rels/workbook.xml.rels", ''.join(rels))
    for i, (_, path) in enumerate(sheets, 1):
        z.writestr(f"xl/worksheets/sheet{i}.xml", sheet_xml(path))
PY
}

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
    "$CC" $OPT_FLAGS $ARCH_FLAGS $INCLUDE_FLAGS -c "$src_s" -o "$obj"
    "$CC" $OPT_FLAGS $ARCH_FLAGS $OMP_FLAGS $INCLUDE_FLAGS "$BENCH" "$obj" -o "$bin"
}

build_source() {
    local label=$1
    local src=$2
    local obj="$WORKDIR/${label}.o"
    local bin="$WORKDIR/bench_${label}"
    "$CC" $OPT_FLAGS $ARCH_FLAGS $INCLUDE_FLAGS -c "$src" -o "$obj"
    "$CC" $OPT_FLAGS $ARCH_FLAGS $OMP_FLAGS $INCLUDE_FLAGS "$BENCH" "$obj" -o "$bin"
}

generate_neon_compute_only() {
    local out_s=$1
    cat > "$out_s" <<'EOF'
// Generated by run_m8_parts.sh: NEON M8 compute-only baseline.
// Keeps bfmmla and loop control; removes zero, A/B load, A reorder, and C store.

#define bf16gemm_k_ld       bf16gemm_neon_unused_ld
#define bf16gemm_k_ld_b     bf16gemm_neon_unused_ld_b
#define bf16gemm_k_ld1      bf16gemm_neon_unused_ld1
#define bf16gemm_k_ld2      bf16gemm_neon_unused_ld2
#define bf16gemm_k_ld4      bf16gemm_neon_unused_ld4
#define bf16gemm_k_ld1_b    bf16gemm_neon_unused_ld1_b
#define bf16gemm_k_ld2_b    bf16gemm_neon_unused_ld2_b
#define bf16gemm_k_ld4_b    bf16gemm_neon_unused_ld4_b
#define bf16gemm_k_nld_b    bf16gemm_neon_unused_nld_b
#define bf16gemm_k_nld1_b   bf16gemm_neon_unused_nld1_b
#define bf16gemm_k_nld2_b   bf16gemm_neon_unused_nld2_b
#define bf16gemm_k_nld4_b   bf16gemm_neon_unused_nld4_b
#include "bf16gemm_k.S"

	.purgem INIT
	.macro INIT
	stp x19, x20, [sp, #-16]!
	stp x21, x22, [sp, #-16]!
	stp x23, x24, [sp, #-16]!
	stp x25, x26, [sp, #-16]!
	stp x27, x28, [sp, #-16]!
	stp x29, x30, [sp, #-16]!
	stp d8,  d9,  [sp, #-16]!
	stp d10, d11, [sp, #-16]!
	stp d12, d13, [sp, #-16]!
	stp d14, d15, [sp, #-16]!
	movi v0.8h, #1
	movi v1.8h, #1
	movi v2.8h, #1
	movi v3.8h, #1
	movi v4.8h, #1
	movi v5.8h, #1
	movi v6.8h, #1
	movi v7.8h, #1
	movi v8.8h, #1
	movi v25.8h, #1
	movi v26.8h, #1
	movi v27.8h, #1
	movi v28.8h, #1
	movi v29.8h, #1
	movi v30.8h, #1
	movi v31.8h, #1
	.endm

	.macro LOAD_NOTHING_8
	.endm

	.macro STORE_NOTHING_8
	.endm

	.purgem LOAD_A0_B0_8
	.purgem COMPUTE_A1_B1_8
	.purgem COMPUTE_A2_B2_8
	.purgem COMPUTE_A3_B3_8
	.purgem LOAD_A0_B0_STRIDED_8
	.purgem COMPUTE_A1_B1_STRIDED_8
	.purgem COMPUTE_A2_B2_STRIDED_8

	.macro COMPUTE_A0_B0_ONLY
	bfmmla v9.4s,  v0.8h, v4.8h
	bfmmla v10.4s, v0.8h, v5.8h
	bfmmla v11.4s, v0.8h, v6.8h
	bfmmla v12.4s, v0.8h, v7.8h
	bfmmla v13.4s, v1.8h, v4.8h
	bfmmla v14.4s, v1.8h, v5.8h
	bfmmla v15.4s, v1.8h, v6.8h
	bfmmla v16.4s, v1.8h, v7.8h
	bfmmla v17.4s, v2.8h, v4.8h
	bfmmla v18.4s, v2.8h, v5.8h
	bfmmla v19.4s, v2.8h, v6.8h
	bfmmla v20.4s, v2.8h, v7.8h
	bfmmla v21.4s, v3.8h, v4.8h
	bfmmla v22.4s, v3.8h, v5.8h
	bfmmla v23.4s, v3.8h, v6.8h
	bfmmla v24.4s, v3.8h, v7.8h
	.endm

	.macro COMPUTE_A1_B1_ONLY
	bfmmla v9.4s,  v8.8h, v28.8h
	bfmmla v10.4s, v8.8h, v29.8h
	bfmmla v11.4s, v8.8h, v30.8h
	bfmmla v12.4s, v8.8h, v31.8h
	bfmmla v13.4s, v25.8h, v28.8h
	bfmmla v14.4s, v25.8h, v29.8h
	bfmmla v15.4s, v25.8h, v30.8h
	bfmmla v16.4s, v25.8h, v31.8h
	bfmmla v17.4s, v26.8h, v28.8h
	bfmmla v18.4s, v26.8h, v29.8h
	bfmmla v19.4s, v26.8h, v30.8h
	bfmmla v20.4s, v26.8h, v31.8h
	bfmmla v21.4s, v27.8h, v28.8h
	bfmmla v22.4s, v27.8h, v29.8h
	bfmmla v23.4s, v27.8h, v30.8h
	bfmmla v24.4s, v27.8h, v31.8h
	.endm

	.macro LOAD_A0_B0_8
	.endm

	.macro COMPUTE_A1_B1_8
	COMPUTE_A0_B0_ONLY
	.endm

	.macro COMPUTE_A2_B2_8
	COMPUTE_A1_B1_ONLY
	.endm

	.macro COMPUTE_A3_B3_8
	COMPUTE_A1_B1_ONLY
	.endm

	.macro LOAD_A0_B0_STRIDED_8
	LOAD_A0_B0_8
	.endm

	.macro COMPUTE_A1_B1_STRIDED_8
	COMPUTE_A1_B1_8
	.endm

	.macro COMPUTE_A2_B2_STRIDED_8
	COMPUTE_A2_B2_8
	.endm

GEMM_BODY bf16gemm_k_nld_f_m8, 8, 8, LOAD_NOTHING_8, STORE_NOTHING_8, 2, 32

	.global bf16gemm_k_nld_b_m8
	.type bf16gemm_k_nld_b_m8, %function
bf16gemm_k_nld_b_m8:
	b bf16gemm_k_nld_f_m8
	.size bf16gemm_k_nld_b_m8, .-bf16gemm_k_nld_b_m8

	.global bf16gemm_k_nld_bias_f_m8
	.type bf16gemm_k_nld_bias_f_m8, %function
bf16gemm_k_nld_bias_f_m8:
	b bf16gemm_k_nld_f_m8
	.size bf16gemm_k_nld_bias_f_m8, .-bf16gemm_k_nld_bias_f_m8
EOF
}

build_neon_compute_only() {
    local src_s="$WORKDIR/neon_compute_only.S"
    local obj="$WORKDIR/neon_compute_only.o"
    local bin="$WORKDIR/bench_neon_compute_only"
    generate_neon_compute_only "$src_s"
    "$CC" $OPT_FLAGS $ARCH_FLAGS $INCLUDE_FLAGS -c "$src_s" -o "$obj"
    "$CC" $OPT_FLAGS $ARCH_FLAGS $OMP_FLAGS $INCLUDE_FLAGS "$BENCH" "$obj" -o "$bin"
}

build_baseline() {
    local impl=$1
    case "$impl" in
        sve)
            build_variant "compute_only"
            cp "$WORKDIR/bench_compute_only" "$WORKDIR/bench_sve_compute_only"
            ;;
        neon)
            build_neon_compute_only
            ;;
        *) echo "unknown baseline impl: $impl" >&2; exit 2 ;;
    esac
}

build_tail_full() {
    local impl=$1
    local bin="$WORKDIR/bench_tail_${impl}"
    case "$impl" in
        sve)
            "$CC" $OPT_FLAGS $ARCH_FLAGS $OMP_FLAGS $INCLUDE_FLAGS "$TAIL_BENCH" \
                "$LIB_DIR/bf16gemm_sve.c" "$LIB_DIR/bf16gemm_sve.S" \
                "$LIB_DIR/i8gemm_sve.c" -o "$bin" -lm
            ;;
        neon)
            "$CC" $OPT_FLAGS $ARCH_FLAGS $OMP_FLAGS $INCLUDE_FLAGS "$TAIL_BENCH" \
                "$LIB_DIR/bf16gemm_mt.c" "$LIB_DIR/bf16gemm_k.S" \
                "$LIB_DIR/bf16gemm_k_bias.S" -o "$bin" -lm
            ;;
        *) echo "unknown tail impl: $impl" >&2; exit 2 ;;
    esac
}

build_dispatch() {
    local impl=$1
    local bin="$WORKDIR/bench_dispatch_${impl}"
    case "$impl" in
        sve)
            "$CC" $OPT_FLAGS $ARCH_FLAGS $OMP_FLAGS $INCLUDE_FLAGS -DBENCH_SVE=1 \
                "$DISPATCH_BENCH" \
                "$LIB_DIR/bf16gemm_sve.c" "$LIB_DIR/bf16gemm_sve.S" \
                "$LIB_DIR/i8gemm_sve.c" "$LIB_DIR/i8gemm_sve.S" \
                -o "$bin" -lm
            ;;
        neon)
            "$CC" $OPT_FLAGS $ARCH_FLAGS $OMP_FLAGS $INCLUDE_FLAGS \
                "$DISPATCH_BENCH" \
                "$LIB_DIR/bf16gemm_mt.c" "$LIB_DIR/bf16gemm_k.S" \
                "$LIB_DIR/bf16gemm_k_bias.S" \
                "$LIB_DIR/i8gemm_mt.c" "$LIB_DIR/i8gemm_k.S" \
                "$LIB_DIR/i8gemm_k_bias.S" \
                -o "$bin" -lm
            ;;
        *) echo "unknown dispatch impl: $impl" >&2; exit 2 ;;
    esac
}

if [[ "$RUN_PARTS" != "0" ]]; then
    for variant in $PART_VARIANTS; do
        build_variant "$variant"
    done
fi

if [[ "$RUN_BASELINES" != "0" ]]; then
    for impl in $BASELINE_IMPLS; do
        build_baseline "$impl"
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

if [[ "$RUN_STRIDE" != "0" ]]; then
    for impl in $STRIDE_IMPLS; do
        case "$impl" in
            sve)  build_source "store_sve" "$SVE_SRC" ;;
            neon) build_source "store_neon" "$NEON_SRC" ;;
            *) echo "unknown stride impl: $impl" >&2; exit 2 ;;
        esac
    done
fi

if [[ "$RUN_BATCH" != "0" ]]; then
    for impl in $BATCH_IMPLS; do
        case "$impl" in
            sve)  build_source "store_sve" "$SVE_SRC" ;;
            neon) build_source "store_neon" "$NEON_SRC" ;;
            *) echo "unknown batch impl: $impl" >&2; exit 2 ;;
        esac
    done
fi

if [[ "$RUN_DISPATCH" != "0" ]]; then
    for impl in $DISPATCH_IMPLS; do
        build_dispatch "$impl"
    done
fi

if [[ "$RUN_TAILS" != "0" ]]; then
    for impl in $TAIL_IMPLS; do
        build_tail_full "$impl"
    done
fi

{
    echo "# core binding: ${CORESET:-none}"
    echo "# available cores: $AVAILABLE_CORES"
    echo "# threads: $THREADS"
    echo "# case mode: $CASE_MODE"
    echo "# parts cases: $(case_count)"
    if [[ "$RUN_TAILS" != "0" ]]; then
        echo "# tail cases: $(tail_case_count)"
    fi
    if [[ "$PRUNE_BIG_CASE_THREADS" != "0" ]]; then
        echo "# prune big-case threads: ABC>${PRUNE_TOTAL_BYTES} bytes requires ceil(ABC/threads)<=${PRUNE_PER_THREAD_BYTES} bytes"
    else
        echo "# prune big-case threads: disabled"
    fi
    echo "# OMP_PLACES=$OMP_PLACES OMP_PROC_BIND=$OMP_PROC_BIND"
    count_total_steps
    echo "# planned benchmark calls: $TOTAL_STEPS"
} >&2

echo "variant,mode,cache,M,K,N,threads,KiB,reps,GFLOPS,pct_of_330,pct_of_330xthreads,stride_factor,batch_count" > "$OUT"

run_parts_case() {
    local M=$1
    local K=$2
    local N=$3
    local T impl variant mode
    for T in $THREADS; do
        if ! case_thread_allowed "$M" "$K" "$N" "$T"; then
            continue
        fi
        if [[ "$RUN_BASELINES" != "0" ]]; then
            for impl in $BASELINE_IMPLS; do
                progress "parts impl=${impl}_compute_only mode=f32 M=$M K=$K N=$N threads=$T"
                run_bench "$WORKDIR/bench_${impl}_compute_only" "${impl}_compute_only" "$M" "$K" "$N" "$REPS" "$WARMUP" "$RUNS" f32 "$T" >> "$OUT"
            done
        fi
        if [[ "$RUN_PARTS" != "0" ]]; then
            for variant in $PART_VARIANTS; do
                progress "parts variant=sve_${variant} mode=f32 M=$M K=$K N=$N threads=$T"
                run_bench "$WORKDIR/bench_${variant}" "sve_${variant}" "$M" "$K" "$N" "$REPS" "$WARMUP" "$RUNS" f32 "$T" >> "$OUT"
            done
        fi
        if [[ "$RUN_STORES" != "0" ]]; then
            for impl in $STORE_IMPLS; do
                for mode in $STORE_MODES; do
                    progress "parts variant=${impl}_${mode} mode=$mode M=$M K=$K N=$N threads=$T"
                    run_bench "$WORKDIR/bench_store_${impl}" "${impl}_${mode}" "$M" "$K" "$N" "$REPS" "$WARMUP" "$RUNS" "$mode" "$T" >> "$OUT"
                done
            done
        fi
    done
}

if [[ "$CASE_MODE" == "grid" ]]; then
    for K in $K_VALUES; do
        for M in $M_VALUES; do
            for N in $N_VALUES; do
                run_parts_case "$M" "$K" "$N"
            done
        done
    done
else
    for case in $CASES; do
        IFS=, read -r M K N <<< "$case"
        run_parts_case "$M" "$K" "$N"
    done
fi

echo "wrote $OUT"

if [[ "$RUN_STRIDE" != "0" ]]; then
    echo "variant,mode,cache,M,K,N,threads,KiB,reps,GFLOPS,pct_of_330,pct_of_330xthreads,stride_factor,batch_count" > "$STRIDE_OUT"

    run_stride_case() {
        local M=$1
        local K=$2
        local N=$3
        local T impl mode sf
        for T in $THREADS; do
            if ! case_thread_allowed "$M" "$K" "$N" "$T"; then
                continue
            fi
            for impl in $STRIDE_IMPLS; do
                for mode in $STRIDE_MODES; do
                    for sf in $STRIDE_FACTORS; do
                        progress "stride impl=$impl mode=$mode stride=$sf M=$M K=$K N=$N threads=$T"
                        run_bench "$WORKDIR/bench_store_${impl}" "${impl}_${mode}_stride${sf}" \
                            "$M" "$K" "$N" "$REPS" "$WARMUP" "$RUNS" "$mode" "$T" "$sf" 1 >> "$STRIDE_OUT"
                    done
                done
            done
        done
    }

    if [[ "$CASE_MODE" == "grid" ]]; then
        for K in $K_VALUES; do
            for M in $M_VALUES; do
                for N in $N_VALUES; do
                    run_stride_case "$M" "$K" "$N"
                done
            done
        done
    else
        for case in $CASES; do
            IFS=, read -r M K N <<< "$case"
            run_stride_case "$M" "$K" "$N"
        done
    fi
    echo "wrote $STRIDE_OUT"
fi

if [[ "$RUN_BATCH" != "0" ]]; then
    echo "variant,mode,cache,M,K,N,threads,KiB,reps,GFLOPS,pct_of_330,pct_of_330xthreads,stride_factor,batch_count" > "$BATCH_OUT"

    run_batch_case() {
        local M=$1
        local K=$2
        local N=$3
        local T impl mode bc
        for T in $THREADS; do
            if ! case_thread_allowed "$M" "$K" "$N" "$T"; then
                continue
            fi
            for impl in $BATCH_IMPLS; do
                for mode in $BATCH_MODES; do
                    for bc in $BATCH_COUNTS; do
                        progress "batch impl=$impl mode=$mode batch=$bc M=$M K=$K N=$N threads=$T"
                        run_bench "$WORKDIR/bench_store_${impl}" "${impl}_${mode}_batch${bc}" \
                            "$M" "$K" "$N" "$REPS" "$WARMUP" "$RUNS" "$mode" "$T" 1 "$bc" >> "$BATCH_OUT"
                    done
                done
            done
        done
    }

    if [[ "$CASE_MODE" == "grid" ]]; then
        for K in $K_VALUES; do
            for M in $M_VALUES; do
                for N in $N_VALUES; do
                    run_batch_case "$M" "$K" "$N"
                done
            done
        done
    else
        for case in $CASES; do
            IFS=, read -r M K N <<< "$case"
            run_batch_case "$M" "$K" "$N"
        done
    fi
    echo "wrote $BATCH_OUT"
fi

if [[ "$RUN_DISPATCH" != "0" ]]; then
    echo "impl,dtype,cache,M,K,N,threads,batch_count,KiB,reps,perf,pct_of_baseline,pct_of_baseline_xthreads" > "$DISPATCH_OUT"

    run_dispatch_case() {
        local M=$1
        local K=$2
        local N=$3
        local T impl dtype bc
        for T in $THREADS; do
            if ! case_thread_allowed "$M" "$K" "$N" "$T"; then
                continue
            fi
            for impl in $DISPATCH_IMPLS; do
                for dtype in $DISPATCH_DTYPES; do
                    for bc in $DISPATCH_BATCH_COUNTS; do
                        progress "dispatch impl=$impl dtype=$dtype batch=$bc M=$M K=$K N=$N threads=$T"
                        run_bench "$WORKDIR/bench_dispatch_${impl}" "$impl" "$dtype" \
                            "$M" "$K" "$N" "$REPS" "$WARMUP" "$RUNS" "$T" "$bc" >> "$DISPATCH_OUT"
                    done
                done
            done
        done
    }

    if [[ "$CASE_MODE" == "grid" ]]; then
        for K in $K_VALUES; do
            for M in $M_VALUES; do
                for N in $N_VALUES; do
                    run_dispatch_case "$M" "$K" "$N"
                done
            done
        done
    else
        for case in $CASES; do
            IFS=, read -r M K N <<< "$case"
            run_dispatch_case "$M" "$K" "$N"
        done
    fi
    echo "wrote $DISPATCH_OUT"
fi

if [[ "$RUN_TAILS" != "0" ]]; then
    echo "impl,mode,cache,base_M,tail_M,K,N,threads,KiB,reps,base_GFLOPS,tail_GFLOPS,tail_vs_base_pct,tail_drop_pct,pct_of_330,pct_of_330xthreads" > "$TAIL_OUT"

    run_tail_case() {
        local BASE_M=$1
        local K=$2
        local N=$3
        local T impl base_line base_gflops D TAIL_M tail_line max_d worst_M
        max_d=$(max_tail_delta)
        worst_M=$((BASE_M + max_d))
        for T in $THREADS; do
            if ! case_thread_allowed "$worst_M" "$K" "$N" "$T"; then
                continue
            fi
            for impl in $TAIL_IMPLS; do
                progress "tail-base impl=$impl M=$BASE_M K=$K N=$N threads=$T"
                base_line=$(run_bench "$WORKDIR/bench_tail_${impl}" "${impl}_full_base" "$BASE_M" "$K" "$N" "$REPS" "$WARMUP" "$RUNS" "$T" "$TAIL_N_ALIGN")
                base_gflops=$(awk -F, '{print $10}' <<< "$base_line")
                for D in $TAIL_DELTAS; do
                    TAIL_M=$((BASE_M + D))
                    progress "tail impl=$impl base_M=$BASE_M tail_M=$TAIL_M K=$K N=$N threads=$T"
                    tail_line=$(run_bench "$WORKDIR/bench_tail_${impl}" "${impl}_full_tail" "$TAIL_M" "$K" "$N" "$REPS" "$WARMUP" "$RUNS" "$T" "$TAIL_N_ALIGN")
                    awk -F, -v impl="$impl" -v base_m="$BASE_M" -v tail_m="$TAIL_M" -v base_g="$base_gflops" '
                        {
                            tail_g = $10 + 0.0
                            rel = (base_g > 0.0) ? tail_g * 100.0 / base_g : 0.0
                            drop = 100.0 - rel
                            printf "%s,%s,%s,%d,%d,%d,%d,%d,%.1f,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                                   impl, $2, $3, base_m, tail_m, $5, $6, $7, $8, $9,
                                   base_g, tail_g, rel, drop, $11, $12
                        }
                    ' <<< "$tail_line" >> "$TAIL_OUT"
                done
            done
        done
    }

    if [[ "$CASE_MODE" == "grid" ]]; then
        for K in $K_VALUES; do
            for BASE_M in $TAIL_BASE_M_VALUES; do
                for N in $N_VALUES; do
                    run_tail_case "$BASE_M" "$K" "$N"
                done
            done
        done
    else
        for case in $TAIL_CASES; do
            IFS=, read -r BASE_M K N <<< "$case"
            run_tail_case "$BASE_M" "$K" "$N"
        done
    fi
    echo "wrote $TAIL_OUT"
fi

if [[ "$WRITE_XLSX" != "0" ]]; then
    XLSX_SHEETS=(parts "$OUT")
    if [[ "$RUN_TAILS" != "0" ]]; then
        XLSX_SHEETS+=(tail "$TAIL_OUT")
    fi
    if [[ "$RUN_STRIDE" != "0" ]]; then
        XLSX_SHEETS+=(stride "$STRIDE_OUT")
    fi
    if [[ "$RUN_BATCH" != "0" ]]; then
        XLSX_SHEETS+=(batch "$BATCH_OUT")
    fi
    if [[ "$RUN_DISPATCH" != "0" ]]; then
        XLSX_SHEETS+=(dispatch "$DISPATCH_OUT")
    fi
    write_xlsx "$RESULTS_XLSX" "${XLSX_SHEETS[@]}"
    echo "wrote $RESULTS_XLSX"
    if [[ "$KEEP_CSV" == "0" ]]; then
        rm -f "$OUT"
        if [[ "$RUN_TAILS" != "0" ]]; then
            rm -f "$TAIL_OUT"
        fi
        if [[ "$RUN_STRIDE" != "0" ]]; then
            rm -f "$STRIDE_OUT"
        fi
        if [[ "$RUN_BATCH" != "0" ]]; then
            rm -f "$BATCH_OUT"
        fi
        if [[ "$RUN_DISPATCH" != "0" ]]; then
            rm -f "$DISPATCH_OUT"
        fi
    fi
fi
