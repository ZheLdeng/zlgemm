// gemm_thread_pool.c — Persistent pthread-based thread pool for GEMM dispatch
//
// Implementation notes:
//   - Each worker receives its thread index (tid) as a startup argument,
//     avoiding a mutex-held tid-scanning step.
//   - Workers sleep on a condition variable between dispatches.
//   - Static tile partitioning: each thread gets contiguous tiles.
//   - A dispatch costs: 1 broadcast + N lock/unlock pairs + 1 done_signal.
//   - Compared to OpenMP: eliminates team creation/destruction overhead
//     at the cost of explicit mutex operations per dispatch.

#include "gemm_thread_pool.h"

#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define GEMM_TP_MAX_THREADS 256

typedef struct {
    gemm_thread_pool_t *pool;
    int tid;
} gemm_tp_worker_arg_t;

struct gemm_thread_pool {
    int nthreads;

    pthread_t threads[GEMM_TP_MAX_THREADS];

    // Synchronization
    pthread_mutex_t mutex;
    pthread_cond_t  start_cond;   // workers wait here for work
    pthread_cond_t  done_cond;    // master waits here for completion
    pthread_cond_t  ready_cond;   // master waits for workers to be ready

    // Work descriptor (set by master before broadcast)
    volatile int   n_tiles;
    gemm_tp_kernel_t kernel;
    volatile void *ctx;

    // Completion / readiness tracking
    volatile int ready_count;     // workers that reached the wait loop
    volatile int completed;       // workers that finished this dispatch
    volatile int generation;      // distinguishes successive dispatches

    volatile bool shutdown;
};

// ── Worker thread routine ──────────────────────────────────────────────

static void *gemm_tp_worker(void *arg) {
    gemm_tp_worker_arg_t *wa = (gemm_tp_worker_arg_t *)arg;
    gemm_thread_pool_t *pool = wa->pool;
    int tid                  = wa->tid;
    int nthreads             = pool->nthreads;
    free(wa);  // argument no longer needed

    // --- Signal ready and enter wait loop ---
    pthread_mutex_lock(&pool->mutex);

    pool->ready_count++;
    if (pool->ready_count == nthreads) {
        pthread_cond_signal(&pool->ready_cond);
    }

    int local_gen = pool->generation;  // 0 initially

    while (1) {
        // Wait for work or shutdown
        while (!pool->shutdown && pool->generation == local_gen) {
            pthread_cond_wait(&pool->start_cond, &pool->mutex);
        }

        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }

        local_gen = pool->generation;

        // Snapshot work params under lock
        int n_tiles         = pool->n_tiles;
        gemm_tp_kernel_t fn = pool->kernel;
        void *ctx            = (void *)pool->ctx;

        pthread_mutex_unlock(&pool->mutex);

        // --- Static tile partitioning ---
        if (fn && n_tiles > 0 && tid < n_tiles) {
            int blocks_per = n_tiles / nthreads;
            int extra      = n_tiles % nthreads;
            int start, end;

            if (tid < extra) {
                start = tid * (blocks_per + 1);
                end   = start + blocks_per + 1;
            } else {
                start = extra * (blocks_per + 1) + (tid - extra) * blocks_per;
                end   = start + blocks_per;
            }

            for (int t = start; t < end; t++) {
                fn(t, tid, ctx);
            }
        }

        // --- Completion barrier ---
        pthread_mutex_lock(&pool->mutex);
        pool->completed++;
        if (pool->completed == nthreads) {
            // Last thread done — wake the master
            pthread_cond_signal(&pool->done_cond);
        }
        // Loop back to wait (mutex still held)
    }

    return NULL;
}

// ── Public API ─────────────────────────────────────────────────────────

gemm_thread_pool_t *gemm_tp_create(int nthreads) {
    if (nthreads < 1)
        nthreads = 1;
    if (nthreads > GEMM_TP_MAX_THREADS)
        nthreads = GEMM_TP_MAX_THREADS;

    gemm_thread_pool_t *pool = (gemm_thread_pool_t *)calloc(1, sizeof(*pool));
    if (!pool)
        return NULL;

    pool->nthreads    = nthreads;
    pool->generation  = 0;
    pool->ready_count = 0;
    pool->shutdown    = false;

    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->start_cond, NULL);
    pthread_cond_init(&pool->done_cond, NULL);
    pthread_cond_init(&pool->ready_cond, NULL);

    // Create worker threads (do NOT hold the mutex, so workers can
    // acquire it and signal readiness).
    for (int i = 0; i < nthreads; i++) {
        gemm_tp_worker_arg_t *arg = (gemm_tp_worker_arg_t *)malloc(sizeof(*arg));
        if (!arg) {
            pool->shutdown = true;
            pool->nthreads = i;
            pthread_mutex_lock(&pool->mutex);
            pool->generation++;
            pthread_cond_broadcast(&pool->start_cond);
            pthread_mutex_unlock(&pool->mutex);
            for (int j = 0; j < i; j++)
                pthread_join(pool->threads[j], NULL);
            pthread_mutex_destroy(&pool->mutex);
            pthread_cond_destroy(&pool->start_cond);
            pthread_cond_destroy(&pool->done_cond);
            pthread_cond_destroy(&pool->ready_cond);
            free(pool);
            return NULL;
        }
        arg->pool = pool;
        arg->tid  = i;

        if (pthread_create(&pool->threads[i], NULL, gemm_tp_worker, arg) != 0) {
            free(arg);
            pool->shutdown = true;
            pool->nthreads = i;
            pthread_mutex_lock(&pool->mutex);
            pool->generation++;
            pthread_cond_broadcast(&pool->start_cond);
            pthread_mutex_unlock(&pool->mutex);
            for (int j = 0; j < i; j++)
                pthread_join(pool->threads[j], NULL);
            pthread_mutex_destroy(&pool->mutex);
            pthread_cond_destroy(&pool->start_cond);
            pthread_cond_destroy(&pool->done_cond);
            pthread_cond_destroy(&pool->ready_cond);
            free(pool);
            return NULL;
        }
    }

    // Wait for all workers to be ready (entered their wait loop)
    pthread_mutex_lock(&pool->mutex);
    while (pool->ready_count < nthreads) {
        pthread_cond_wait(&pool->ready_cond, &pool->mutex);
    }

    // Fire a dummy dispatch so every worker confirms the protocol works
    // and is parked on start_cond before we return.
    pool->n_tiles   = 0;
    pool->kernel    = NULL;
    pool->ctx       = NULL;
    pool->completed = 0;
    pool->generation = 1;
    pthread_cond_broadcast(&pool->start_cond);

    while (pool->completed < pool->nthreads) {
        pthread_cond_wait(&pool->done_cond, &pool->mutex);
    }
    pthread_mutex_unlock(&pool->mutex);

    return pool;
}

void gemm_tp_destroy(gemm_thread_pool_t *tp) {
    if (!tp)
        return;

    pthread_mutex_lock(&tp->mutex);
    tp->shutdown = true;
    tp->generation++;
    pthread_cond_broadcast(&tp->start_cond);
    pthread_mutex_unlock(&tp->mutex);

    for (int i = 0; i < tp->nthreads; i++) {
        pthread_join(tp->threads[i], NULL);
    }

    pthread_mutex_destroy(&tp->mutex);
    pthread_cond_destroy(&tp->start_cond);
    pthread_cond_destroy(&tp->done_cond);
    pthread_cond_destroy(&tp->ready_cond);
    free(tp);
}

int gemm_tp_num_threads(const gemm_thread_pool_t *tp) {
    return tp ? tp->nthreads : 0;
}

void gemm_tp_run_tiles(gemm_thread_pool_t *tp, int n_tiles,
                       gemm_tp_kernel_t kernel, void *ctx) {
    if (!tp || n_tiles <= 0 || !kernel)
        return;

    pthread_mutex_lock(&tp->mutex);

    tp->n_tiles   = n_tiles;
    tp->kernel    = kernel;
    tp->ctx       = ctx;
    tp->completed = 0;
    tp->generation++;

    // Wake all workers
    pthread_cond_broadcast(&tp->start_cond);

    // Wait for all workers to complete
    while (tp->completed < tp->nthreads) {
        pthread_cond_wait(&tp->done_cond, &tp->mutex);
    }

    pthread_mutex_unlock(&tp->mutex);
}
