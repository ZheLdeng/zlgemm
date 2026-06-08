# i8gemm / BF16 GEMM Workspace

This directory is organized into three working areas:

| Directory | Purpose |
| --- | --- |
| `lib/` | Production headers, C dispatchers, assembly kernels, and library Makefile |
| `tests/` | Correctness tests, benchmark drivers, scripts, and test-only kernels |
| `results/` | Reports and selected benchmark result artifacts |

## Build Libraries

```bash
make -C lib
```

Default library outputs:

- `lib/build/libi8gemm_sve.a`
- `lib/build/libi8gemm_sve.so`

Legacy NEON-compatible outputs are built separately because they export the
same public API symbols:

```bash
make -C lib neon
```

## Build And Run Tests

```bash
make -C tests
make -C tests test-sve
make -C tests test-neon
```

The root Makefile delegates to these folders:

```bash
make        # build lib + tests
make test   # run SVE and NEON correctness tests
make clean
```

## M8 Benchmark Script

```bash
cd tests
THREADS=auto c=0-79 RESULTS_XLSX=../results/m8/m8_80c_shape_sweep.xlsx ./run_m8_parts.sh
```

The script now defaults to writing CSV/XLSX under `results/m8/`.
