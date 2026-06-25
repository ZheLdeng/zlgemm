// i8gemm_msplit.h -- M-split-focused multithreaded I8 GEMM (C = A * B, int32).
//
// A companion to the i8gemm_sve.c dispatch. Where that path may subdivide the
// output along N and use 2D thread grids, this lib ONLY partitions the output
// by M (contiguous row-bands, full N per thread). When M is too small to fill
// the requested thread count by M-banding alone, it keeps every thread busy by
// splitting the K reduction (split-K) over the efficient 8-row tile, then sums
// the partial products. Bit-exact with the reference (i32 add is associative).
//
// B packing layout is identical to i8_pack_B (from i8gemm_sve.c); reuse the same
// B_reo buffer. Build: link i8gemm_msplit.c + i8gemm_msplit_k.S together with the
// i8_pack_B definition (i8gemm_sve.c) or the standalone packer here.

#ifndef I8GEMM_MSPLIT_H
#define I8GEMM_MSPLIT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int8_t  i8_t;
typedef int32_t i32_t;

// Core dispatch (zero-allocation in the M-split case; the split-K case uses a
// thread-local cached scratch for the partials). A: M x K_r row-major int8;
// B_reo: pre-packed by i8_pack_B (K_r x N_r); C: M x N_r row-major int32,
// zero-init NOT required (kernel writes every element it owns).
// K_r must be a multiple of 16, N_r a multiple of the n-tile (VL/2).
void i8gemm_msplit_dispatch(const i8_t *A, const i8_t *B_reo,
                            i32_t *C, int M, int K_r, int N_r,
                            int nthreads);

// Convenience wrapper: pads K/N, packs B internally, handles arbitrary K,N.
// A_orig: M x K, B_orig: K x N row-major int8; C: M x N row-major int32.
void i8gemm_msplit(const i8_t *A_orig, const i8_t *B_orig,
                   i32_t *C, int M, int K, int N, int nthreads);

#ifdef __cplusplus
}
#endif

#endif // I8GEMM_MSPLIT_H
