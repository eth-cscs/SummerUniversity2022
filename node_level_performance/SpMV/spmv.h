#pragma once

#ifdef __NVCC__
#include "cuda_runtime.h"
#endif


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <tuple>
#include <typeindex>
#include <unordered_map>

#include <time.h>

#ifdef USE_LIKWID
#   define LIKWID_PERFMON
#   define LIKWID_NVMON
#   include <likwid-marker.h>

// When compiling with nvcc assume we want to measure GPU counters. Else we
// use the markers for CPUs.

#   ifdef __NVCC__
#       define MARKER_INIT()          LIKWID_NVMARKER_INIT
#       define MARKER_DEINIT()        LIKWID_NVMARKER_CLOSE

#       define MARKER_START(name)     LIKWID_NVMARKER_START(name)
#       define MARKER_STOP(name)      LIKWID_NVMARKER_STOP(name)

#       define MARKER_REGISTER(name)  LIKWID_NVMARKER_REGISTER(name)
#       define MARKER_RESET(name)     LIKWID_NVMARKER_RESET(name)
#   else
#       define MARKER_INIT()          LIKWID_MARKER_INIT
#       define MARKER_DEINIT()        LIKWID_MARKER_CLOSE

#       define MARKER_START(name)     LIKWID_MARKER_START(name)
#       define MARKER_STOP(name)      LIKWID_MARKER_STOP(name)

#       define MARKER_REGISTER(name)  LIKWID_MARKER_REGISTER(name)
#       define MARKER_RESET(name)     LIKWID_MARKER_RESET(name)
#   endif

#else

#   define MARKER_INIT()
#   define MARKER_DEINIT()

#   define MARKER_START(name)
#   define MARKER_STOP(name)

#   define MARKER_REGISTER(name)
#   define MARKER_RESET(name)
#endif

// 2097152 b = 2 * 1024 * 1024 b = 2 MiB
#define DEFAULT_ALIGNMENT		2097152

#define RESTRICT				__restrict__

#ifdef __NVCC__
// No. of threads per block.
constexpr int default_block_size = 512;
#endif

using ST = long;
using ST = ST;

enum class MatrixFormat { Csr, EllRm, EllCm, SellCSigma };


class Kernel
{
public:

    template <typename VT, typename IT>
    using fn_csr_t = void(*)(
        const ST num_rows,
        const IT * RESTRICT row_ptrs,
        const IT * RESTRICT col_idxs,
        const VT * RESTRICT values,
        const VT * RESTRICT x,
        VT * RESTRICT y);

    template <typename VT, typename IT>
    using fn_ell_t = void(*)(
        const ST num_rows,
        const ST n_els_per_row,
        const IT * RESTRICT col_idxs,
        const VT * RESTRICT values,
        const VT * RESTRICT x,
        VT * RESTRICT y);

    // TODO: adjust to real sell-c-sigma parameters.
    template <typename VT, typename IT>
    using fn_scs_t = void(*)(
                 const ST C,
                 const ST n_chunks,
                 const IT * RESTRICT chunk_ptrs,
                 const IT * RESTRICT chunk_lengths,
                 const IT * RESTRICT col_idxs,
                 const VT * RESTRICT values,
                 const VT * RESTRICT x,
                 VT * RESTRICT y);

    using fn_void_t = void (*)();

    struct entry_t
    {
        entry_t() = default;

        entry_t(fn_void_t kernel, bool is_gpu_kernel, MatrixFormat format)
        : void_kernel(kernel), is_gpu_kernel(is_gpu_kernel), format(format)
        {}

        template <typename VT, typename IT>
        fn_csr_t<VT, IT> as_csr_kernel() const { return (fn_csr_t<VT, IT>)void_kernel; }

        template <typename VT, typename IT>
        fn_ell_t<VT, IT> as_ell_kernel() const { return (fn_ell_t<VT, IT>)void_kernel; }

        template <typename VT, typename IT>
        fn_scs_t<VT, IT> as_scs_kernel() const { return (fn_scs_t<VT, IT>)void_kernel; }

        fn_void_t void_kernel{};
        bool is_gpu_kernel { false };
        MatrixFormat format   { MatrixFormat::Csr };
    };

    using type_index_pair_t = std::tuple<std::type_index, std::type_index>;

    using kernel_types_t = std::map<type_index_pair_t, entry_t>;

    using kernels_t = std::unordered_map<std::string, kernel_types_t>;

    template <typename VT, typename IT>
    static fn_void_t add(std::string name,
                         std::type_index value_type,
                         std::type_index index_type,
                         fn_csr_t<VT, IT> kernel,
                         bool is_gpu,
                         MatrixFormat format)
    {
        return add(name, value_type, index_type, (fn_void_t)kernel, is_gpu, format);
    }

    template <typename VT, typename IT>
    static fn_void_t add(std::string name,
                         std::type_index value_type,
                         std::type_index index_type,
                         fn_ell_t<VT, IT> kernel,
                         bool is_gpu,
                         MatrixFormat format)
    {
        return add(name, value_type, index_type, (fn_void_t)kernel, is_gpu, format);
    }

    template <typename VT, typename IT>
    static fn_void_t add(std::string name,
                         std::type_index value_type,
                         std::type_index index_type,
                         fn_scs_t<VT, IT> kernel,
                         bool is_gpu,
                         MatrixFormat format)
    {
        return add(name, value_type, index_type, (fn_void_t)kernel, is_gpu, format);
    }

    static kernels_t & kernels();

private:

    static fn_void_t add(std::string name,
                         std::type_index value_type,
                         std::type_index index_type,
                         fn_void_t kernel,
                         bool is_gpu,
                         MatrixFormat format);

};

#define REG_KERNEL_(name, id, line, kernel, t1, t2, is_gpu, format) \
    static auto kernel_func_ ## id ## _ ## line = Kernel::add( \
        name, \
        std::type_index(typeid(t1)), std::type_index(typeid(t2)), \
        kernel<t1, t2>, \
        is_gpu, format)

#define REG_KERNEL(name, id, line, kernel, t1, t2, is_gpu, format) \
    REG_KERNEL_(name, id, line, kernel, t1, t2, is_gpu, format)

#define REG_KERNELS(name, kernel, is_gpu, format) \
    REG_KERNEL(name, 1, __LINE__, kernel, float,  int, is_gpu, format); \
    REG_KERNEL(name, 2, __LINE__, kernel, double, int, is_gpu, format)

/**
 * Return wall time in seconds.
 */
static double inline get_time()
{
    timespec tp;
    int err;
    err = clock_gettime(CLOCK_MONOTONIC, &tp);

    if (err != 0) {
        std::fprintf(stderr, "ERROR: clock_gettime failed with error %d: %s\n",
                     err, strerror(err));
        std::exit(EXIT_FAILURE);
    }

    return tp.tv_sec + tp.tv_nsec * 1e-9;
}

void log(const char * format, ...);

#ifdef __NVCC__

#define assert_gpu(expr) assert_gpu_impl((expr), __FILE__, __LINE__)

static void inline
assert_gpu_impl(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess) {
        printf("assert in %s:%d failed with %d.\n", file, line, error);
        exit(EXIT_FAILURE);
    }
}

#endif // __NVCC__
