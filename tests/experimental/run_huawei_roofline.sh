#!/bin/bash
# Huawei 80-core smmla compute-ceiling diagnostic.
# Builds the accumulator-count roofline and runs it pinned to one core, then
# (if perf is available) collects the counter breakdown that tells us whether
# the SVE smmla ceiling is throughput-capped, latency-bound, or stall-bound.
#
# Usage:  bash run_huawei_roofline.sh [CORE] [GHZ]
#   CORE: core id to pin to (default 80 -- a Huawei big core; use 0 on V1)
#   GHZ : core frequency for smmla/cycle (default 2.9 Huawei; 2.6 V1)
set -e
cd "$(dirname "$0")"
CORE=${1:-80}
GHZ=${2:-2.9}
mkdir -p build
cc -O3 -mcpu=native -o build/huawei_roofline huawei_roofline.c huawei_roofline.S

echo "==================================================================="
echo " smmla accumulator-count roofline (pinned to core $CORE, GHz=$GHZ)"
echo " interpret: smmla/cycle flat across acc8..acc30  => throughput-capped"
echo "            rises with acc count                 => latency-bound"
echo "==================================================================="
taskset -c "$CORE" ./build/huawei_roofline "$GHZ"

echo
echo "==================================================================="
echo " perf counter breakdown on the 24-acc SVE loop (if perf available)"
echo "==================================================================="
EVENTS="cycles,instructions"
# Add backend-stall + SVE events if the kernel exposes them (names vary).
for e in stall_backend stall_backend_mem sve_inst_spec ase_sve_int_spec; do
  if perf list 2>/dev/null | grep -qw "$e"; then EVENTS="$EVENTS,$e"; fi
done
echo "events: $EVENTS"
# isolate the 24-acc SVE loop by a long run; perf measures the whole process but
# the loop dominates (warmup + 6x40M iters).
perf stat -e "$EVENTS" taskset -c "$CORE" ./build/huawei_roofline "$GHZ" 2>&1 | \
  grep -E "cycles|instructions|stall|sve|spec|seconds|insn per" || \
  echo "(perf unavailable -- run the roofline numbers above; smmla/cycle alone is decisive)"
