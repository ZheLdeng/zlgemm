// bench_acl_dispatch.cpp -- ACL CpuGemmAssemblyDispatch benchmark adapter.
//
// Input layout: A is MxK row-major, B is KxN row-major, C/D is MxN
// row-major from the user's perspective. ACL TensorShape(K,M), TensorShape(N,K)
// uses the same contiguous row-major memory order for those matrices.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <string>
#include <vector>

#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/Tensor.h"
#include "src/cpu/kernels/assembly/arm_gemm/arm_gemm.hpp"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/operators/internal/CpuGemmAssemblyDispatch.h"

using namespace arm_compute;
using namespace arm_compute::cpu;

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static bool env_flag(const char *name, bool fallback) {
    const char *s = std::getenv(name);
    if (!s || !*s) return fallback;
    return std::atoi(s) != 0;
}

static size_t round64(size_t n) {
    return (n + 63u) & ~(size_t)63u;
}

static void zero_tensor(Tensor &t) {
    std::memset(t.buffer(), 0, t.info()->total_size());
}

static float value_f32(size_t i, int salt) {
    return (float)(((int)((i + (size_t)salt) % 17) - 8)) * 0.125f;
}

template <typename T>
static void fill_raw(Tensor &t, size_t count, int salt);

template <>
void fill_raw<float>(Tensor &t, size_t count, int salt) {
    auto *p = reinterpret_cast<float *>(t.buffer());
    for (size_t i = 0; i < count; ++i) p[i] = value_f32(i, salt);
}

template <>
void fill_raw<int8_t>(Tensor &t, size_t count, int salt) {
    auto *p = reinterpret_cast<int8_t *>(t.buffer());
    for (size_t i = 0; i < count; ++i) p[i] = (int8_t)(((int)((i + (size_t)salt) % 17) - 8));
}

static uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    u += ((u >> 16) & 1u) + 0x7fffu;
    return (uint16_t)(u >> 16);
}

static void fill_bf16(Tensor &t, size_t count, int salt) {
    auto *p = reinterpret_cast<uint16_t *>(t.buffer());
    for (size_t i = 0; i < count; ++i) p[i] = f32_to_bf16(value_f32(i, salt));
}

static const char *cache_class(double kib) {
    if (kib <= 96.0) return "L1";
    if (kib <= 640.0) return "H2";
    if (kib <= 1280.0) return "L2";
    return "GT_L2";
}

static void print_row(const char *impl, const char *dtype, int M, int K, int N,
                      int threads, double kib, int reps, double perf,
                      const char *status, const char *note) {
    std::printf("%s,%s,%s,%d,%d,%d,%d,%.1f,%d,%.3f,%s,%s\n",
                impl, dtype, cache_class(kib), M, K, N, threads, kib, reps,
                perf, status, note);
}

static std::string acl_kernel_note(const char *dtype, int M, int K, int N, int threads) {
    const CPUInfo &ci = NEScheduler::get().cpu_info();
    arm_gemm::Activation act;
    arm_gemm::GemmConfig cfg;
    arm_gemm::GemmArgs args(&ci, (unsigned int)M, (unsigned int)N, (unsigned int)K, 1, 1, 1, false, act, threads,
                            false, std::strcmp(dtype, "bf16") == 0, false, &cfg);

    std::vector<arm_gemm::KernelDescription> kernels;
    if (std::strcmp(dtype, "bf16") == 0) {
        kernels = arm_gemm::get_compatible_kernels<arm_compute::bfloat16, arm_compute::bfloat16, float,
                                                   arm_gemm::Nothing>(args, {});
    } else {
        kernels = arm_gemm::get_compatible_kernels<int8_t, int8_t, int32_t, arm_gemm::Nothing>(args, {});
    }

    std::string note = "selected=";
    bool found_default = false;
    for (const auto &kdesc : kernels) {
        if (kdesc.is_default) {
            note += kdesc.name;
            found_default = true;
            break;
        }
    }
    if (!found_default) {
        note += kernels.empty() ? "none" : kernels.front().name;
    }
    note += "; compatible=";
    for (size_t i = 0; i < kernels.size(); ++i) {
        if (i) note += "|";
        note += kernels[i].name;
    }
    return note;
}

static bool bench_one(const char *dtype, int M, int K, int N, int reps,
                      int warmup, int runs, int threads, double *perf_out,
                      double *kib_out, const char **note_out) {
    Tensor a;
    Tensor b;
    Tensor d;
    Tensor c;
    bool use_c = false;

    DataType a_dt = DataType::UNKNOWN;
    DataType b_dt = DataType::UNKNOWN;
    DataType d_dt = DataType::UNKNOWN;

    if (std::strcmp(dtype, "bf16") == 0) {
        a_dt = DataType::BFLOAT16;
        b_dt = DataType::BFLOAT16;
        d_dt = DataType::F32;
        *kib_out = ((double)M * K * 2.0 + (double)K * N * 2.0 +
                    (double)M * N * 4.0) /
                   1024.0;
        static std::string note;
        note = acl_kernel_note(dtype, M, K, N, threads);
        *note_out = note.c_str();
    } else if (std::strcmp(dtype, "i8") == 0) {
        a_dt = DataType::QASYMM8_SIGNED;
        b_dt = DataType::QASYMM8_SIGNED;
        d_dt = DataType::S32;
        use_c = true;
        *kib_out = ((double)M * K + (double)K * N + (double)M * N * 4.0) /
                   1024.0;
        static std::string note;
        note = acl_kernel_note(dtype, M, K, N, threads);
        *note_out = note.c_str();
    } else {
        return false;
    }

    a.allocator()->init(TensorInfo(TensorShape((size_t)K, (size_t)M), 1, a_dt));
    b.allocator()->init(TensorInfo(TensorShape((size_t)N, (size_t)K), 1, b_dt));
    d.allocator()->init(TensorInfo(TensorShape((size_t)N, (size_t)M), 1, d_dt));
    if (use_c) {
        c.allocator()->init(TensorInfo(TensorShape((size_t)N), 1, DataType::S32));
    }

    AsmGemmInfo info;
    info.method = AsmConvMethod::Im2Col;
    info.reshape_b_only_on_first_run = true;
    info.accumulate = false;
    info.fast_mode = std::strcmp(dtype, "bf16") == 0;

    ITensorInfo *c_info = use_c ? c.info() : nullptr;
    Status st = CpuGemmAssemblyDispatch::validate(a.info(), b.info(), c_info, d.info(), info);
    if (!bool(st)) {
        *note_out = "ACL validate rejected shape/dtype";
        return false;
    }
    a.allocator()->allocate();
    b.allocator()->allocate();
    d.allocator()->allocate();
    if (use_c) c.allocator()->allocate();

    if (std::strcmp(dtype, "bf16") == 0) {
        fill_bf16(a, (size_t)M * (size_t)K, 1);
        fill_bf16(b, (size_t)K * (size_t)N, 2);
    } else {
        fill_raw<int8_t>(a, (size_t)M * (size_t)K, 1);
        fill_raw<int8_t>(b, (size_t)K * (size_t)N, 2);
        zero_tensor(c);
    }
    zero_tensor(d);

    ITensorPack run_pack{{ACL_SRC_0, &a}, {ACL_SRC_1, &b}, {ACL_DST, &d}};
    ITensorPack prep_pack{{ACL_SRC_1, &b}};
    if (use_c) {
        run_pack.add_tensor(ACL_SRC_2, &c);
        prep_pack.add_tensor(ACL_SRC_2, &c);
    }

    NEScheduler::get().set_num_threads((unsigned int)threads);

    if (warmup > 0) {
        CpuGemmAssemblyDispatch warmup_gemm;
        warmup_gemm.configure(a.info(), b.info(), c_info, d.info(), info);
        MemoryGroup warmup_memory_group;
        auto warmup_workspace =
            manage_workspace<Tensor>(warmup_gemm.workspace(), warmup_memory_group,
                                     run_pack, prep_pack);
        for (int i = 0; i < warmup; ++i) {
            MemoryGroupResourceScope scope(warmup_memory_group);
            warmup_gemm.prepare(prep_pack);
            warmup_gemm.run(run_pack);
        }
        (void)warmup_workspace;
    }

    double best = 0.0;
    const double ops = 2.0 * (double)M * (double)K * (double)N;
    const bool prepare_in_timing = env_flag("ACL_PREPARE_IN_TIMING", true);
    for (int r = 0; r < runs; ++r) {
        CpuGemmAssemblyDispatch gemm;
        gemm.configure(a.info(), b.info(), c_info, d.info(), info);
        MemoryGroup run_memory_group;
        auto run_workspace =
            manage_workspace<Tensor>(gemm.workspace(), run_memory_group,
                                     run_pack, prep_pack);
        if (!prepare_in_timing) {
            MemoryGroupResourceScope scope(run_memory_group);
            gemm.prepare(prep_pack);
        }
        double t0 = now_sec();
        if (prepare_in_timing) {
            MemoryGroupResourceScope scope(run_memory_group);
            gemm.prepare(prep_pack);
        }
        for (int i = 0; i < reps; ++i) {
            MemoryGroupResourceScope scope(run_memory_group);
            gemm.run(run_pack);
        }
        double dt = (now_sec() - t0) / (double)reps;
        best = std::max(best, ops / dt / 1e9);
        (void)run_workspace;
    }

    *perf_out = best;
    return true;
}

int main(int argc, char **argv) {
    if (argc != 9) {
        std::fprintf(stderr, "usage: %s bf16|i8 M K N reps warmup runs threads\n", argv[0]);
        return 2;
    }
    const char *dtype = argv[1];
    int M = std::atoi(argv[2]);
    int K = std::atoi(argv[3]);
    int N = std::atoi(argv[4]);
    int reps = std::atoi(argv[5]);
    int warmup = std::atoi(argv[6]);
    int runs = std::atoi(argv[7]);
    int threads = std::atoi(argv[8]);

    double perf = 0.0, kib = 0.0;
    const char *note = "";
    bool ok = bench_one(dtype, M, K, N, reps, warmup, runs, threads, &perf, &kib, &note);
    print_row("acl_auto", dtype, M, K, N, threads, kib, reps, perf,
              ok ? "ok" : "unsupported", note);
    return ok ? 0 : 0;
}
