// gemm_thread_pool.h — Persistent thread pool for GEMM dispatch
//
// Provides a pthread-based thread pool that avoids the per-call thread
// creation/destruction overhead of OpenMP parallel regions. Threads are
// created once and reused across calls.
//
// Key design decisions:
//   - Static tile partitioning (same as OpenMP schedule(static)) —
//     optimal for uniform GEMM tiles where every tile has equal compute.
//   - Single condition-variable wakeup per dispatch, single barrier at end.
//   - Zero internal allocation per call (all buffers passed via ctx pointer).
//
// Usage:
//   gemm_thread_pool_t *tp = gemm_tp_create(8);
//   gemm_tp_run_tiles(tp, n_tiles, my_kernel, &ctx);
//   gemm_tp_destroy(tp);

#ifndef GEMM_THREAD_POOL_H
#define GEMM_THREAD_POOL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef struct gemm_thread_pool gemm_thread_pool_t;

// Create a thread pool with nthreads persistent workers.
// Returns NULL on failure. nthreads is clamped to [1, 256].
gemm_thread_pool_t *gemm_tp_create(int nthreads);

// Destroy the thread pool, joining all workers.
// Safe to call with NULL (no-op).
void gemm_tp_destroy(gemm_thread_pool_t *tp);

// Return the number of threads in this pool.
int gemm_tp_num_threads(const gemm_thread_pool_t *tp);

// Kernel function signature: called per tile with tile index, thread id,
// and opaque context pointer.
typedef void (*gemm_tp_kernel_t)(int tile, int tid, void *ctx);

// Dispatch work: partition [0, n_tiles) statically across threads.
// Each thread calls kernel(tile, tid, ctx) for every tile in its range.
// Returns after all threads complete. Safe to call concurrently from
// a single master thread (not reentrant from multiple callers).
void gemm_tp_run_tiles(gemm_thread_pool_t *tp, int n_tiles,
                       gemm_tp_kernel_t kernel, void *ctx);

#ifdef __cplusplus
}
#endif

#endif // GEMM_THREAD_POOL_H
