#include "spmv.h"
#include "mtx-reader.h"
#include "vectors.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <sstream>
#include <tuple>
#include <typeinfo>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// The default C to use for sell-c-sigma, when no C is specified.
enum { SCS_DEFAULT_C = 8 };

// Initialize all matrices and vectors the same.
// Use -rand to initialize randomly.
static bool g_same_seed_for_every_vector = true;

// Log information.
static bool g_log = false;

template <typename VT, typename IT> using V = Vector<VT, IT>;
template <typename VT, typename IT> using VG = VectorGpu<VT, IT>;

template <typename T> struct max_rel_error {};

template <> struct max_rel_error<float> {
    using base_value_type = float;
    constexpr static float value = 1e-5f;
};
template <> struct max_rel_error<double> {
    using base_value_type = double;
    constexpr static double value = 1e-13;
};
template <> struct max_rel_error<std::complex<float>> {
    using base_value_type = float;
    constexpr static float  value = 1e-5f;
};
template <> struct max_rel_error<std::complex<double>> {
    using base_value_type = double;
    constexpr static double value = 1e-13;
};


struct Config
{
    long n_els_per_row { -1 };      // ell
    long chunk_size    { -1 };      // sell-c-sigma
    long sigma         { -1 };      // sell-c-sigma

    // Initialize rhs vector with random numbers.
    bool random_init_x { true };
    // Override values of the matrix, read via mtx file, with random numbers.
    bool random_init_A { false };

    // No. of repetitions to perform. 0 for automatic detection.
    unsigned long n_repetitions{};

    // Verify result of SpVM.
    bool verify_result{ true };

    // Verify result against solution of COO kernel.
    bool verify_result_with_coo{ false };

    // Print incorrect elements from solution.
    bool verbose_verification{ true };

    // Sort rows/columns of sparse matrix before
    // converting it to a specific format.
    bool sort_matrix{ true };
};


template <typename VT, typename IT>
struct DefaultValues
{
    VT A{ 2.0};
    VT x{ 1.01};
    VT y{ 0.12};

    VT * x_values{};
    ST n_x_values{};

    VT * y_values{};
    ST n_y_values{};
};


template <typename VT, typename IT>
struct ScsData
{
    ST C{};
    ST sigma{};

    ST n_rows{};
    ST n_cols{};
    ST n_rows_padded{};
    ST n_chunks{};
    ST n_elements{};            // No. of nz + padding.
    ST nnz{};                   // No. of nz only.

    V<IT, IT> chunk_ptrs;       // Chunk start offsets into col_idxs & values.
    V<IT, IT> chunk_lengths;    // Length of one row in a chunk.
    V<IT, IT> col_idxs;
    V<VT, IT> values;
    V<IT, IT> old_to_new_idx;
};


struct BenchmarkResult {
    double perf_gflops{};
    double mem_mb{};

    unsigned int size_value_type{};
    unsigned int size_index_type{};

    unsigned long n_calls{};
    unsigned long n_total_calls{};
    double duration_total_s{};
    double duration_kernel_s{};

    bool is_result_valid{false};
    std::string notes;

    std::string value_type_str;
    std::string index_type_str;

    uint64_t value_type_size{};
    uint64_t index_type_size{};

    ST n_rows{};
    ST n_cols{};
    ST nnz{};

    double fill_in_percent{};
    long C{};
    long sigma{};
    long nzr{};

    bool was_matrix_sorted{false};

    double mem_m_mb{};
    double mem_x_mb{};
    double mem_y_mb{};

    double beta{};

    double cb_a_0{};
    double cb_a_nzc{};
};


Kernel::fn_void_t
Kernel::add(std::string name,
            std::type_index value_type,
            std::type_index index_type,
            fn_void_t kernel,
            bool is_gpu,
            MatrixFormat format)
{
    kernels_t & ks = kernels();

    if (ks.find(name) == ks.end()) {
        ks.insert({name, kernel_types_t()});
    }

    ks[name].insert(std::make_pair(
                        std::make_tuple(value_type, index_type),
                        entry_t(kernel, is_gpu, format)
                    ));

    return kernel;
}


Kernel::kernels_t &
Kernel::kernels()
{
    static kernels_t ks;
    return ks;
}


void
log(const char * format, ...)
{
    if (g_log) {
        static double log_started = get_time();

        va_list args;
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), "# [%10.4f] %s", get_time() - log_started, format);

        va_start(args, format);
        vprintf(buffer, args);
        va_end(args);

        fflush(stdout);
    }
}


template <typename VT>
static void
print_vector(const std::string & name,
             const VT * begin,
             const VT * end)
{
    std::cout << name << " [" << end - begin << "]:";
    for (const VT * it = begin; it != end; ++it) {
        std::cout << " " << *it;
    }
    std::cout << "\n";
}


template <typename VT, typename IT>
static void
print_vector(const std::string & name,
             const V<VT, IT> & v)
{
    print_vector(name, v.data(), v.data() + v.n_rows);
}


template <typename T,
          typename std::enable_if<
                 std::is_integral<T>::value
              && std::is_signed<T>::value,
              bool
          >::type = true >
static bool
will_add_overflow(T a, T b)
{
    if (a > 0 && b > 0) {
        return std::numeric_limits<T>::max() - a < b;
    }
    else if (a < 0 && b < 0) {
        return std::numeric_limits<T>::min() - a > b;
    }

    return false;
}


template <typename T,
          typename std::enable_if<
                 std::is_integral<T>::value
              && std::is_unsigned<T>::value,
              bool
          >::type = true >
static bool
will_add_overflow(T a, T b)
{
    return std::numeric_limits<T>::max() - a < b;
}


template <typename T,
          typename std::enable_if<
                 std::is_integral<T>::value
              && std::is_signed<T>::value,
              bool
          >::type = true >
static bool
will_mult_overflow(T a, T b)
{
    if (a == 0 || b == 0) {
            return false;
    }
    else if (a < 0 && b > 0 ) {
        return std::numeric_limits<T>::min() / b > a;
    }
    else if (a > 0 && b < 0) {
        return std::numeric_limits<T>::min() / a > b;
    }
    else if (a > 0 && b > 0) {
        return std::numeric_limits<T>::max() / a < b;
    }
    else {
        T difference =
                  std::numeric_limits<T>::max()
                + std::numeric_limits<T>::min();

        if (difference == 0) { // symmetric case
            return std::numeric_limits<T>::min() / a < b * T{-1};
        }
        else { // abs(min) > max
            T c = std::numeric_limits<T>::min() - difference;

            if (a < c || b < c) return true;

            T ap = a * T{-1};
            T bp = b * T{-1};

            return std::numeric_limits<T>::max() / ap < bp;
        }
    }
}


template <typename T,
          typename std::enable_if<
                 std::is_integral<T>::value
              && std::is_unsigned<T>::value,
              bool
          >::type = true >
static bool
will_mult_overflow(T a, T b)
{
    if (a == 0 || b == 0) {
        return false;
    }

    return std::numeric_limits<T>::max() / a < b;
}


static std::tuple<std::string, uint64_t>
type_info_from_type_index(const std::type_index & ti)
{
    static std::unordered_map<std::type_index, std::tuple<std::string, uint64_t>> type_map = {
        { std::type_index(typeid(double))       , std::make_tuple( "dp"       , sizeof(double)        ) },
        { std::type_index(typeid(float))        , std::make_tuple( "sp"       , sizeof(float)         ) },

        { std::type_index(typeid(int))          , std::make_tuple( "int"      , sizeof(int)           ) },
        { std::type_index(typeid(long))         , std::make_tuple( "long"     , sizeof(long)          ) },
        { std::type_index(typeid(int32_t))      , std::make_tuple( "int32_t"  , sizeof(int32_t)       ) },
        { std::type_index(typeid(int64_t))      , std::make_tuple( "int64_t"  , sizeof(int64_t)       ) },

        { std::type_index(typeid(unsigned int)) , std::make_tuple( "uint"     , sizeof(unsigned int)  ) },
        { std::type_index(typeid(unsigned long)), std::make_tuple( "ulong"    , sizeof(unsigned long) ) },
        { std::type_index(typeid(uint32_t))     , std::make_tuple( "uint32_t" , sizeof(uint32_t)      ) },
        { std::type_index(typeid(uint64_t))     , std::make_tuple( "uint64_t" , sizeof(uint64_t)      ) }
    };


    auto it = type_map.find(ti);

    if (it == type_map.end()) {
        return std::make_tuple(std::string{"unknown"}, uint64_t{0});
    }

    return it->second;
}

static std::string
type_name_from_type_index(const std::type_index & ti)
{
    return std::get<0>(type_info_from_type_index(ti));
}

#if 0
// Currently unused
static uint64_t
type_size_from_type_index(const std::type_index & ti)
{
    return std::get<1>(type_info_from_type_index(ti));
}
#endif

template <typename T>
static std::string
type_name_from_type()
{
    return type_name_from_type_index(std::type_index(typeid(T)));
}


class Histogram
{
public:

    Histogram()
        : Histogram(29)
    {}

    Histogram(size_t n_buckets)
    {
        bucket_upper_bounds().push_back(0);

        int start = 1;
        int end   = start * 10;
        int inc   = start;

        while (n_buckets > 0) {

            for (int i = start; i < end && n_buckets > 0; i += inc, --n_buckets) {
                bucket_upper_bounds().push_back(i);
            }

            start = end;
            end   = start * 10;
            inc   = start;

            // if (n_buckets < 9) {
            //     n_buckets = 0;
            // }
            // else {
            //     n_buckets -= 9;
            // }
        }

        bucket_upper_bounds().push_back(end + 1);
        // for (int i = 1; i < 10; ++i) {
        //     bucket_upper_bounds().push_back(i);
        // }
        // for (int i = 10; i < 100; i += 10) {
        //     bucket_upper_bounds().push_back(i);
        // }
        // for (int i = 100; i < 1000; i += 100) {
        //     bucket_upper_bounds().push_back(i);
        // }
        // bucket_upper_bounds().push_back(1000000);
        bucket_counts().resize(bucket_upper_bounds().size());
    }

    // Option -Ofast optimizes too much here in some cases, so revert to
    // no optimization.
    // (pow gets replaced by exp( ln(10.0) * bp ) with ln(10.0) as a constant)

    void
    __attribute__((optimize("O0")))
    insert(int64_t value) {
        if (value < 0) {
        // if (value < 1) {
            throw std::invalid_argument("value must be > 0");
        }

        // remove if 0 should not be counted

        if (value == 0) {
            bucket_counts()[0] += 1;
            return;
        }
        else if (value == 1) {
            bucket_counts()[1] += 1;
            return;
        }

        value -= 1;

        double x = std::log10(static_cast<double>(value));
        double bp = std::floor(x);

        size_t inner_index = (size_t)std::floor(static_cast<double>(value) / std::pow(10.0, bp));
        size_t outer_index = 9 * (size_t)(bp);

        // decrement by 1 (-1) when value == 0 should not be counted!

        size_t index = outer_index + inner_index; // TODO: - 1ul;
        // shift index, as we manually insert 0.
        index += 1;

        if (index >= bucket_counts().size()) {
            index = bucket_counts().size() - 1ul;
        }

        bucket_counts()[index] += 1;
    }

    std::vector<uint64_t> & bucket_counts()      { return bucket_counts_; }
    std::vector<int64_t> & bucket_upper_bounds() { return bucket_upper_bounds_; }

    const std::vector<uint64_t> & bucket_counts()      const { return bucket_counts_; }
    const std::vector<int64_t> & bucket_upper_bounds() const { return bucket_upper_bounds_; }

private:
    std::vector<uint64_t> bucket_counts_;
    std::vector<int64_t>  bucket_upper_bounds_;

};


template <typename T = double>
struct Statistics
{
    T min{std::numeric_limits<T>::max()};
    T max{std::numeric_limits<T>::min()};
    T avg{};
    T std_dev{};
    T cv{};
    T median{};

    Histogram hist{};
};

template <typename T = double>
struct RowOrColStats
{
    T value_min{std::numeric_limits<T>::max()};
    T value_max{std::numeric_limits<T>::min()};
    int n_values{};
    int n_non_zeros{};
    uint64_t min_idx{std::numeric_limits<uint64_t>::max()};
    uint64_t max_idx{};
};

template <typename T = double>
struct MatrixStats
{
    std::vector<RowOrColStats<T>> all_rows;
    std::vector<RowOrColStats<T>> all_cols;

    Statistics<T> row_lengths{};
    Statistics<T> col_lengths{};
    T densitiy{};

    uint64_t n_rows{};
    uint64_t n_cols{};
    uint64_t nnz{};

    Statistics<T> bandwidths;

    bool is_symmetric{};
    bool is_sorted{};
};

template <typename T = double, typename U = double, typename F>
static struct Statistics<T>
get_statistics(
        std::vector<U> & entries,
        F & getter,
        size_t n_buckets_in_histogram = 29)
{
    Statistics<T> s{};
    s.hist = Histogram(n_buckets_in_histogram);
    T sum{};

    // min/max/avg

    for (const auto & entry : entries) {
        T n_values = getter(entry);

        sum += n_values;

        if (s.max < n_values) {
            s.max = n_values;
        }
        if (s.min > n_values) {
            s.min = n_values;
        }
    }

    s.avg = sum / (T)entries.size();

    // std deviation, cv

    T sum_squares{};

    for (const auto & entry : entries) {
        T n_values = (T)getter(entry);
        sum_squares += (n_values - s.avg) * (n_values - s.avg);
    }

    s.std_dev = std::sqrt(sum_squares / (T)entries.size());
    s.cv = s.std_dev / s.avg;

    // WARNING: entries will be sorted...
    auto median = [&getter](std::vector<U> & entries) -> int
    {
        auto middle_el = begin(entries) + entries.size() / 2;
        std::nth_element(begin(entries), middle_el, end(entries),
                         [&getter](const auto & a, const auto & b) {
                            return getter(a) < getter(b);
                         });

        return getter(entries[entries.size() / 2]);
    };

    s.median = median(entries);

    // Fill histogram

    for (const auto & entry : entries) {
        s.hist.insert(getter(entry));
    }

    return s;
}


template <typename VT, typename IT>
static MatrixStats<double>
get_matrix_stats(const MtxData<VT, IT> & mtx)
{
    MatrixStats<double> stats;
    auto & all_rows = stats.all_rows;
    auto & all_cols = stats.all_cols;

    all_rows.resize(mtx.n_rows);
    all_cols.resize(mtx.n_cols);

    for (ST i = 0; i < mtx.nnz; ++i) {
        IT row = mtx.I[i];
        IT col = mtx.J[i];

        if (mtx.values[i] != VT{}) {
            // Skip if we do not want information about the spread of the values.
            if (all_rows[row].value_min > fabs(mtx.values[i])) {
                all_rows[row].value_min = fabs(mtx.values[i]);
            }

            if (all_rows[row].value_max < fabs(mtx.values[i])) {
                all_rows[row].value_max = fabs(mtx.values[i]);
            }

            if (all_cols[col].value_min > fabs(mtx.values[i])) {
                all_cols[col].value_min = fabs(mtx.values[i]);
            }

            if (all_cols[col].value_max < fabs(mtx.values[i])) {
                all_cols[col].value_max = fabs(mtx.values[i]);
            }
            // Skip end

            ++all_rows[row].n_non_zeros;
            ++all_cols[col].n_non_zeros;
        }

        ++all_rows[row].n_values;
        ++all_cols[col].n_values;

        if ((uint64_t)col > all_rows[row].max_idx) {
            all_rows[row].max_idx = col;
        }
        if ((uint64_t)col < all_rows[row].min_idx) {
            all_rows[row].min_idx = col;
        }

        if ((uint64_t)row > all_rows[col].max_idx) {
            all_rows[col].max_idx = row;
        }
        if ((uint64_t)row < all_rows[col].min_idx) {
            all_rows[col].min_idx = row;
        }
    }

    // compute bandwidth and histogram for bandwidth from row stats
    {
        std::vector<uint64_t> bandwidths;
        bandwidths.reserve(mtx.n_rows);

        for (uint64_t row_idx = 0; row_idx < (uint64_t)stats.all_rows.size(); ++row_idx) {
            const auto & row = stats.all_rows[row_idx];

            uint64_t local_bw = 1;

            if (row_idx > row.min_idx)
                local_bw += row_idx - row.min_idx;
            if (row_idx < row.max_idx)
                local_bw += row.max_idx - row_idx;

            bandwidths.push_back(local_bw);
        }

        auto get_el = [](const uint64_t & e) { return (double)e; };

        // determine needed no. of buckets in histogram
        size_t n_buckets = std::ceil(std::log10(mtx.n_cols) * 9.0 + 1.0);

        stats.bandwidths = get_statistics<double>(bandwidths, get_el, n_buckets);

    }

    auto get_n_values  = [](const RowOrColStats<double> & e) { return e.n_values; };

    stats.row_lengths  = get_statistics<double>(stats.all_rows, get_n_values);
    stats.col_lengths  = get_statistics<double>(stats.all_cols, get_n_values);

    stats.densitiy     = (double)mtx.nnz / ((double)mtx.n_rows * (double)mtx.n_cols);
    stats.n_rows       = mtx.n_rows;
    stats.n_cols       = mtx.n_cols;
    stats.nnz          = mtx.nnz;
    stats.is_symmetric = mtx.is_symmetric;
    stats.is_sorted    = mtx.is_sorted;
    return stats;
}


template <typename IT>
static void
convert_idxs_to_ptrs(const std::vector<IT> &idxs,
                     V<IT, IT> &ptrs)
{
    std::fill(ptrs.data(), ptrs.data() + ptrs.n_rows, 0);

    for (const auto idx : idxs) {
        if (idx + 1 < ptrs.n_rows) {
            ++ptrs[idx + 1];
        }
    }

    std::partial_sum(ptrs.data(), ptrs.data() + ptrs.n_rows, ptrs.data());
}


template <typename VT, typename IT>
static void
convert_to_csr(const MtxData<VT, IT> &mtx,
               V<IT, IT> &row_ptrs,
               V<IT, IT> &col_idxs,
               V<VT, IT> &values)
{
    values = V<VT, IT>(mtx.nnz);
    col_idxs = V<IT, IT>(mtx.nnz);
    row_ptrs = V<IT, IT>(mtx.n_rows + 1);

    std::vector<IT> col_offset_in_row(mtx.n_rows);

    convert_idxs_to_ptrs(mtx.I, row_ptrs);

    for (ST i = 0; i < mtx.nnz; ++i) {
        IT row = mtx.I[i];

        IT idx = row_ptrs[row] + col_offset_in_row[row];

        col_idxs[idx] = mtx.J[i];
        values[idx]   = mtx.values[i];

        col_offset_in_row[row]++;
    }
}


/**
 * Compute maximum number of elements in a row.
 * \p num_rows: Number of rows.
 * \p nnz: Number of non-zeros, also number of elements in \p row_indices.
 * \p row_indices: Array with row indices.
 */
template <typename IT>
static IT
calculate_max_nnz_per_row(
        ST num_rows, ST nnz,
        const IT *row_indices)
{
    std::vector<IT> rptr(num_rows + 1);

    for (ST i = 0; i < nnz; ++i) {
        IT row = row_indices[i];
        if (row + 1 < num_rows) {
            ++rptr[row + 1];
        }
    }

    return *std::max_element(rptr.begin(), rptr.end());
}


/**
 * Create data structures for ELL format from \p mtx.
 *
 * \param col_major If true, column major layout for data structures will
 *                  be used.  If false, row major layout will be used.
 */
template <typename VT, typename IT>
static bool
convert_to_ell(const MtxData<VT, IT> &mtx,
               bool col_major,
               ST &n_els_per_row,
               V<IT, IT> &col_idxs,
               V<VT, IT> &values)
{
    const ST max_els_per_row = calculate_max_nnz_per_row(
                mtx.n_rows, mtx.nnz, mtx.I.data());

    if (n_els_per_row == -1) {
        n_els_per_row = max_els_per_row;
    }
    else {
        if (n_els_per_row < max_els_per_row) {
            fprintf(stderr,
                    "ERROR: ell format: number of elements per row must be >= %ld.\n",
                    (long)max_els_per_row);
            exit(EXIT_FAILURE);
        }
    }

    if (will_mult_overflow(mtx.n_rows, n_els_per_row)) {
        fprintf(stderr, "ERROR: for ELL format no. of padded elements will exceed size type.\n");
        return false;
    }

    const ST n_ell_elements = mtx.n_rows * n_els_per_row;

    values   = V<VT, IT>(n_ell_elements);
    col_idxs = V<IT, IT>(n_ell_elements);

    for (ST i = 0; i < n_ell_elements; ++i) {
        values[i]   = VT{};
        col_idxs[i] = IT{};
    }

    std::vector<IT> col_idx_in_row(mtx.n_rows);

    if (col_major) {
        for (ST i = 0; i < mtx.nnz; ++i) {
            IT row = mtx.I[i];
            IT idx = col_idx_in_row[row] * mtx.n_rows + row;

            col_idxs[idx] = mtx.J[i];
            values[idx]   = mtx.values[i];

            col_idx_in_row[row]++;
        }
    }
    else { /* row major */
        for (ST i = 0; i < mtx.nnz; ++i) {
            IT row = mtx.I[i];
            IT idx = row * max_els_per_row + col_idx_in_row[row];

            col_idxs[idx] = mtx.J[i];
            values[idx]   = mtx.values[i];

            col_idx_in_row[row]++;
        }
    }
    return true;
}


/**
 * Convert \p mtx to sell-c-sigma data structures.
 *
 * If \p C is < 1 then SCS_DEFAULT_C is used.
 * If \p sigma is < 1 then 1 is used.
 *
 * Note: the matrix entries in \p mtx don't need to be sorted.
 */
template <typename VT, typename IT>
static bool
convert_to_scs(const MtxData<VT, IT> & mtx,
               ST C, ST sigma,
               ScsData<VT, IT> & d)
{
    d.nnz    = mtx.nnz;
    d.n_rows = mtx.n_rows;
    d.n_cols = mtx.n_cols;

    d.C = C;
    d.sigma = sigma;

    if (d.C     < 1) { d.C = SCS_DEFAULT_C; }
    if (d.sigma < 1) { d.sigma = 1; }

    if (d.sigma % d.C != 0 && d.sigma != 1) {
        fprintf(stderr, "NOTE: sigma is not a multiple of C\n");
    }

    if (will_add_overflow(d.n_rows, d.C)) {
        fprintf(stderr, "ERROR: no. of padded row exceeds size type.\n");
        return false;
    }
    d.n_chunks      = (mtx.n_rows + d.C - 1) / d.C;

    if (will_mult_overflow(d.n_chunks, d.C)) {
        fprintf(stderr, "ERROR: no. of padded row exceeds size type.\n");
        return false;
    }
    d.n_rows_padded = d.n_chunks * d.C;

    // first enty: original row index
    // second entry: population count of row
    using index_and_els_per_row = std::pair<ST, ST>;

    std::vector<index_and_els_per_row> n_els_per_row(d.n_rows_padded);

    for (ST i = 0; i < d.n_rows_padded; ++i) {
        n_els_per_row[i].first = i;
    }

    for (ST i = 0; i < mtx.nnz; ++i) {
        ++n_els_per_row[mtx.I[i]].second;
    }

    // sort rows in the scope of sigma

    if (will_add_overflow(d.n_rows_padded, d.sigma)) {
        fprintf(stderr, "ERROR: no. of padded rows + sigma exceeds size type.\n");
        return false;
    }

    for (ST i = 0; i < d.n_rows_padded; i += d.sigma) {
        auto begin = &n_els_per_row[i];
        auto end   = (i + d.sigma) < d.n_rows_padded
                        ? &n_els_per_row[i + d.sigma]
                        : &n_els_per_row[d.n_rows_padded];

        std::sort(begin, end,
                  // sort longer rows first
                  [](const auto & a, const auto & b) {
                    return a.second > b.second;
                  });
    }

    // determine chunk_ptrs and chunk_lengths

    // TODO: check chunk_ptrs can overflow

    d.chunk_lengths = V<IT, IT>(d.n_chunks);
    d.chunk_ptrs    = V<IT, IT>(d.n_chunks + 1);

    IT cur_chunk_ptr = 0;

    for (ST i = 0; i < d.n_chunks; ++i) {
        auto begin = &n_els_per_row[i * d.C];
        auto end   = &n_els_per_row[i * d.C + d.C];

        d.chunk_lengths[i] =
                std::max_element(begin, end,
                    [](const auto & a, const auto & b) {
                        return a.second < b.second;
                    })->second;

        if (will_add_overflow(cur_chunk_ptr, d.chunk_lengths[i] * (IT)d.C)) {
            fprintf(stderr, "ERROR: chunck_ptrs exceed index type.\n");
            return false;
        }

        d.chunk_ptrs[i] = cur_chunk_ptr;
        cur_chunk_ptr += d.chunk_lengths[i] * d.C;
    }

    ST n_scs_elements = d.chunk_ptrs[d.n_chunks - 1]
                        + d.chunk_lengths[d.n_chunks - 1] * d.C;
    d.chunk_ptrs[d.n_chunks] = n_scs_elements;

    // construct permutation vector

    d.old_to_new_idx = V<IT, IT>(d.n_rows);

    for (ST i = 0; i < d.n_rows_padded; ++i) {
        IT old_row_idx = n_els_per_row[i].first;

        if (old_row_idx < d.n_rows) {
            d.old_to_new_idx[old_row_idx] = i;
        }
    }

    d.values   = V<VT, IT>(n_scs_elements);
    d.col_idxs = V<IT, IT>(n_scs_elements);

    for (ST i = 0; i < n_scs_elements; ++i) {
        d.values[i]   = VT{};
        d.col_idxs[i] = IT{};
    }

    std::vector<IT> col_idx_in_row(d.n_rows_padded);

    // fill values and col_idxs
    for (ST i = 0; i < d.nnz; ++i) {
        IT row_old = mtx.I[i];
        IT row = d.old_to_new_idx[row_old];

        ST chunk_index = row / d.C;

        IT chunk_start = d.chunk_ptrs[chunk_index];
        IT chunk_row   = row % d.C;

        IT idx = chunk_start + col_idx_in_row[row] * d.C + chunk_row;

        d.col_idxs[idx] = mtx.J[i];
        d.values[idx]   = mtx.values[i];

        col_idx_in_row[row]++;
    }

    d.n_elements = n_scs_elements;

    return true;
}


/**
 * Compare vectors \p reference and \p actual.  Return the no. of elements that
 * differ.
 */
template <typename VT>
static ST
compare_arrays(const VT *reference,
               const VT *actual, const ST n,
               const bool verbose,
               const VT max_rel_error,
               VT & max_rel_error_found)
{
    ST error_counter = 0;
    max_rel_error_found = VT{};

    for (ST i = 0; i < n; ++i) {
        VT rel_error = std::abs((actual[i] - reference[i]) / reference[i]);

        if (rel_error > max_rel_error) {
            if (verbose && error_counter < 10) {
                std::fprintf(stderr,
                             "  expected element %2ld = %19.12e, but got %19.13e, rel. err. %19.12e\n",
                             (long)i, reference[i], actual[i], rel_error);
            }
            ++error_counter;
        }

        if (max_rel_error_found < rel_error) {
            max_rel_error_found = rel_error;
        }
    }

    if (verbose && error_counter > 0) {
        std::fprintf(stderr, "  %ld/%ld elements do not match\n", (long)error_counter,
                     (long)n);
    }

    return error_counter;
}


template <typename VT>
static bool
spmv_verify(
        const VT * y_ref,
        const VT * y_actual,
        const ST n,
        bool verbose)
{
    VT max_rel_error_found{};

    ST error_counter =
            compare_arrays(
                y_ref, y_actual, n,
                verbose,
                max_rel_error<VT>::value,
                max_rel_error_found);

    if (error_counter > 0) {
        // TODO: fix reported name and sizes.
        fprintf(stderr,
                "WARNING: spmv kernel %s (fp size %lu, idx size %lu) is incorrect, "
                "relative error > %e for %ld/%ld elements. Max found rel error %e.\n",
                "", sizeof(VT), 0ul,
                max_rel_error<VT>::value,
                (long)error_counter, (long)n,
                max_rel_error_found);
    }

    return error_counter == 0;
}


template <typename VT, typename IT>
static bool
spmv_verify(const std::string & kernel_name,
            const MtxData<VT, IT> & mtx,
            const std::vector<VT> & x,
            const std::vector<VT> & y_actual)
{
    std::vector<VT> y_ref(mtx.n_rows);

    ST nnz = mtx.nnz;
    if (mtx.I.size() != mtx.J.size() || mtx.I.size() != mtx.values.size()) {
        fprintf(stderr, "ERROR: %s:%d sizes of rows, cols, and values differ.\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    for (ST i = 0; i < nnz; ++i) {
        y_ref[mtx.I[i]] += mtx.values[i] * x[mtx.J[i]];
    }

    return spmv_verify(y_ref.data(), y_actual.data(),
                       y_actual.size(), /*verbose*/ true);
}


template <typename VT, typename IT, typename FN>
static BenchmarkResult
spmv(FN && kernel, bool is_gpu_kernel, const Config & config)
{
    log("running kernel begin\n");

    log("warmup begin\n");

    BenchmarkResult result;

    kernel();

    ++result.n_total_calls;

    if (is_gpu_kernel) {
#ifdef __NVCC__
        assert_gpu(cudaDeviceSynchronize());
#endif
    }

    log("warmup end\n");

    double t_kernel_start = 0.0;
    double t_kernel_end   = 0.0;
    double duration       = 0.0;

    // Indicate if result is invalid, e.g., duration of >1s was not reached.
    bool is_result_valid = true;

    unsigned long n_repetitions = config.n_repetitions > 0 ? config.n_repetitions : 5;
    int repeate_measurement;

    do {
        log("running kernel with %ld repetitions\n", n_repetitions);
        repeate_measurement = 0;

        t_kernel_start = get_time();

        for (unsigned long r = 0; r < n_repetitions; ++r) {
            kernel();
        }

        if (is_gpu_kernel) {
#ifdef __NVCC__
            assert_gpu(cudaDeviceSynchronize());
#endif
        }

        t_kernel_end = get_time();

        duration = t_kernel_end - t_kernel_start;

        result.n_total_calls += n_repetitions;

        if (duration < 1.0 && config.n_repetitions == 0) {
            unsigned long prev_n_repetitions = n_repetitions;
            n_repetitions = std::ceil(n_repetitions / duration * 1.1);

            if (prev_n_repetitions == n_repetitions) {
                ++n_repetitions;
            }

            if (n_repetitions < prev_n_repetitions) {
                // This typically happens if type ulong is too small to hold the
                // number of repetitions we would need for a duration > 1s.
                // We use the time we measured and flag the result.
                log("cannot increase no. of repetitions any further to reach a duration of >1s\n");
                log("aborting measurement for this kernel\n");
                n_repetitions = prev_n_repetitions;
                repeate_measurement = 0;
                is_result_valid = false;
            }
            else {
                repeate_measurement = 1;
            }
        }
    } while (repeate_measurement);

    result.is_result_valid  = is_result_valid;
    result.n_calls          = n_repetitions;
    result.duration_total_s = duration;

    log("running kernel end\n");

    return result;
}


template <typename T, typename DIST, typename ENGINE>
struct random_number
{
    static T
    get(DIST & dist, ENGINE & engine)
    {
        return dist(engine);
    }
};

template <typename DIST, typename ENGINE>
struct random_number<std::complex<float>, DIST, ENGINE>
{
    static std::complex<float>
    get(DIST & dist, ENGINE & engine)
    {
        return std::complex<float>(dist(engine), dist(engine));
    }
};

template <typename DIST, typename ENGINE>
struct random_number<std::complex<double>, DIST, ENGINE>
{
    static std::complex<double>
    get(DIST & dist, ENGINE & engine)
    {
        return std::complex<double>(dist(engine), dist(engine));
    }
};

template <typename VT>
static void
random_init(VT * begin, VT * end)
{
    std::mt19937 engine;

    if (!g_same_seed_for_every_vector) {
        std::random_device rnd_device;
        engine.seed(rnd_device());
    }

    std::uniform_real_distribution<double> dist(0.1, 2.0);

    for (VT * it = begin; it != end; ++it) {
        *it = random_number<VT, decltype(dist), decltype(engine)>::get(dist, engine);
    }
}

template <typename VT, typename IT>
static void
random_init(V<VT, IT> & v)
{
    random_init(v.data(), v.data() + v.n_rows);
}


template <typename VT, typename IT>
static void
init_with_ptr_or_value(V<VT, IT> & x,
                       ST n_x,
                       const std::vector<VT> * x_in,
                       VT default_value,
                       bool init_with_random_numbers = false)
{
    if (!init_with_random_numbers) {
        if (x_in) {
            if (x_in->size() != size_t(n_x)) {
                fprintf(stderr, "ERROR: x_in has incorrect size.\n");
                exit(EXIT_FAILURE);
            }

            for (ST i = 0; i < n_x; ++i) {
                x[i] = (*x_in)[i];
            }
        }
        else {
            for (ST i = 0; i < n_x; ++i) {
                x[i] = default_value;
            }
        }
    }
    else {
        random_init(x);
    }
}


/**
 * Kernel used to compute a reference CSR solution.
 */
template <typename VT, typename IT>
static void
spmv_csr_reference(
        const ST num_rows,
        const IT *row_ptrs,
        const IT *col_idxs,
        const VT *mat_values,
        const VT *x,
        VT *y)
{
    for (ST row = 0; row < num_rows; ++row) {
        VT sum{};
        for (ST k = row_ptrs[row]; k < row_ptrs[row + 1]; ++k) {
            auto val = mat_values[k];
            auto col = col_idxs[k];
            sum += val * x[col];
        }
        y[row] = sum;
    }
}


/**
 * Kernel used to compute a reference ELL solution.
 */
template <typename VT, typename IT>
static void
spmv_ell_reference(
        bool is_col_major,
        const ST num_rows,
        const ST nelems_per_row,
        const IT *col_idxs,
        const VT *mat_values,
        const VT *x,
        VT *y)
{

    if (is_col_major) {
        for (ST row = 0; row < num_rows; row++) {
            VT sum{};
            for (ST i = 0; i < nelems_per_row; i++) {
                auto val = mat_values[row + i * num_rows];
                auto col = col_idxs[  row + i * num_rows];
                sum += val * x[col];
            }
            y[row] = sum;
        }
    }
    else {
        for (ST row = 0; row < num_rows; row++) {
            VT sum{};
            for (ST i = 0; i < nelems_per_row; i++) {
                auto val = mat_values[row * nelems_per_row + i];
                auto col = col_idxs[  row * nelems_per_row + i];
                sum += val * x[col];
            }
            y[row] = sum;
        }
   }
}

/**
 * Kernel for Sell-C-Sigma. Supports all Cs > 0.
 */
template <typename VT, typename IT>
static void
spmv_scs_reference(const ST C,
             const ST n_chunks,
             const IT *chunk_ptrs,
             const IT *chunk_lengths,
             const IT *col_idxs,
             const VT *values,
             const VT *x,
             VT *y)
{

    #pragma omp parallel for schedule(static)
    for (ST c = 0; c < n_chunks; ++c) {
        VT tmp[C];
        for (ST i = 0; i < C; ++i) {
            tmp[i] = VT{};
        }

        IT cs = chunk_ptrs[c];

        for (ST j = 0; j < chunk_lengths[c]; ++j) {
            for (ST i = 0; i < C; ++i) {
                tmp[i] += values[cs + i] * x[col_idxs[cs + i]];
            }
            cs += C;
        }

        for (ST i = 0; i < C; ++i) {
            y[c * C + i] = tmp[i];
        }
    }
}


/**
 * @brief Compute the code balance for a certain MatrixFormat.
 * @param format
 * @param alpha
 * @param beta
 * @param value_type_size
 * @param index_type_size
 * @param nnz
 * @param n_rows
 * @param C_for_scs
 * @param write_allocate_for_lhs
 * @return
 */
static double
code_balance(
    MatrixFormat format,
    double alpha,
    double beta,
    uint64_t value_type_size,
    uint64_t index_type_size,
    uint64_t nnz,
    uint64_t n_rows,
    uint64_t C_for_scs,
    bool write_allocate_for_lhs,
    bool value_type_is_complex)
{
    double accesses_to_lhs = 1.0;

    if (write_allocate_for_lhs)
        accesses_to_lhs += 1.0;

    double v_per_nnz_b{};
    double nzr = (double)nnz / (double)n_rows;

    // just to shorten the formulas.
    const double s_vt = value_type_size;
    const double s_it = index_type_size;

    switch (format) {
    case MatrixFormat::Csr:
    {
        v_per_nnz_b = (s_vt + s_it) + alpha * s_vt
                      + (accesses_to_lhs * s_vt + s_it) / nzr;
        break;
    }
    case MatrixFormat::EllCm:
    case MatrixFormat::EllRm:
    {
        v_per_nnz_b = (s_vt + s_it) / beta + alpha * s_vt
                      + accesses_to_lhs * s_vt / nzr;
        break;
    }
    case MatrixFormat::SellCSigma:
    {
        const double C = C_for_scs;
        const double n_chunks = std::ceil((double)n_rows / C);
        const double n_rows_padded = C * n_chunks;
        const double nzr_padded = (double)nnz / n_rows_padded;

        v_per_nnz_b = (s_vt + s_it) / beta + alpha * s_vt
                      + accesses_to_lhs * s_vt / nzr_padded
                      + 2 * s_it * n_chunks / (double)nnz;
        break;
    }
    default:
        fprintf(stderr, "ERROR: Cannot compute code balance for unknown matrix format.\n");
        exit(EXIT_FAILURE);
    }

    double flops_per_nnz = value_type_is_complex ? 8.0 : 2.0;

    return v_per_nnz_b / flops_per_nnz;
}

static void
compute_code_balances(
    MatrixFormat format,
    bool is_gpu_kernel,
    bool is_vt_complex,
    BenchmarkResult & r)
{
    r.beta = 1.0 / ( r.fill_in_percent / 100.0 + 1.0);

    r.cb_a_0 = code_balance(format,
                            /* alpha */ 0.0,
                            r.beta,
                            r.value_type_size,
                            r.index_type_size,
                            r.nnz,
                            r.n_rows,
                            r.C,
                            true, // y = y + A b -> always WA for LHS
                            is_vt_complex
                            );

    r.cb_a_nzc = code_balance(format,
                            /* alpha */ 1.0 / ((double)r.nnz / (double)r.n_rows),
                            r.beta,
                            r.value_type_size,
                            r.index_type_size,
                            r.nnz,
                            r.n_rows,
                            r.C,
                            true, // y = y + A b -> always WA for LHS
                            is_vt_complex
                            );
}


template <typename VT, typename IT>
static BenchmarkResult
bench_spmv_csr(
                const Config & config,
                const MtxData<VT, IT> &mtx,

                const Kernel::entry_t & k_entry,

                DefaultValues<VT, IT> & defaults,
                std::vector<VT> &x_out,
                std::vector<VT> &y_out,

                const std::vector<VT> * x_in = nullptr)
{
    BenchmarkResult r;

    const ST nnz    = mtx.nnz;
    const ST n_rows = mtx.n_rows;
    const ST n_cols = mtx.n_cols;

    V<VT, IT> values;
    V<IT, IT> col_idxs;
    V<IT, IT> row_ptrs;

    log("converting to csr format start\n");

    convert_to_csr<VT, IT>(mtx, row_ptrs, col_idxs, values);

    log("converting to csr format end\n");

    V<VT, IT> x_csr = V<VT, IT>(mtx.n_cols);
    init_with_ptr_or_value(x_csr, x_csr.n_rows, x_in,
                           defaults.x, config.random_init_x);

    V<VT, IT> y_csr = V<VT, IT>(mtx.n_rows);
    std::uninitialized_fill_n(y_csr.data(), y_csr.n_rows, defaults.y);

    Kernel::fn_csr_t<VT, IT> kernel = k_entry.as_csr_kernel<VT, IT>();

    // print_vector("x", x_csr.data(), x_csr.data() + x_csr.n_rows);
    // print_vector("y(pre)", y_csr.data(), y_csr.data() + y_csr.n_rows);
    //
    // print_vector("row_ptrs", row_ptrs.data(), row_ptrs.data() + row_ptrs.n_rows);
    // print_vector("col_idxs", col_idxs.data(), col_idxs.data() + col_idxs.n_rows);
    // print_vector("values",   values.data(),   values.data()   + values.n_rows);

    if (k_entry.is_gpu_kernel) {
#ifdef __NVCC__
        log("init GPU matrices start\n");
        VG<VT, IT> values_gpu(values);
        VG<IT, IT> col_idxs_gpu(col_idxs);
        VG<IT, IT> row_ptrs_gpu(row_ptrs);
        VG<VT, IT> x_gpu(x_csr);
        VG<VT, IT> y_gpu(y_csr);
        log("init GPU matrices end\n");

        r = spmv<VT, IT>([&]() {
                const int num_blocks = (n_rows + default_block_size - 1) \
                                        / default_block_size;

                kernel<<<num_blocks, default_block_size>>>(n_rows,
                       row_ptrs_gpu.data(), col_idxs_gpu.data(), values_gpu.data(),
                       x_gpu.data(), y_gpu.data());
            },
            /* is_gpu_kernel */ true,
            config);

        y_csr = y_gpu.copy_from_device();
#endif
    }
    else {
        r = spmv<VT, IT>([&]() {
                kernel(n_rows,
                       row_ptrs.data(), col_idxs.data(), values.data(),
                       x_csr.data(), y_csr.data());
            },
            /* is_gpu_kernel */ false,
            config);
    }

    // print_vector("y", y_csr.data(), y_csr.data() + y_csr.n_rows);

    if (config.verify_result) {
        V<VT, IT> y_ref(y_csr.n_rows);
        std::uninitialized_fill_n(y_ref.data(), y_ref.n_rows, defaults.y);

        spmv_csr_reference(n_rows, row_ptrs.data(), col_idxs.data(),
                           values.data(), x_csr.data(), y_ref.data());

        r.is_result_valid &= spmv_verify(y_csr.data(), y_ref.data(),
                                         y_csr.n_rows, config.verbose_verification);
    }

    double mem_matrix_b =
              (double)sizeof(VT) * nnz           // values
            + (double)sizeof(IT) * nnz           // col idxs
            + (double)sizeof(IT) * (n_rows + 1); // row ptrs
    double mem_x_b = (double)sizeof(VT) * n_cols;
    double mem_y_b = (double)sizeof(VT) * n_rows;
    double mem_b   = mem_matrix_b + mem_x_b + mem_y_b;

    r.mem_mb   = mem_b / 1e6;
    r.mem_m_mb = mem_matrix_b / 1e6;
    r.mem_x_mb  = mem_x_b / 1e6;
    r.mem_y_mb  = mem_y_b / 1e6;

    r.n_rows = mtx.n_rows;
    r.n_cols = mtx.n_cols;
    r.nnz    = nnz;

    r.duration_kernel_s = r.duration_total_s / r.n_calls;
    r.perf_gflops       = (double)nnz * 2.0
                          / r.duration_kernel_s
                          / 1e9;                 // Only count usefull flops

    r.value_type_str = type_name_from_type<VT>();
    r.index_type_str = type_name_from_type<IT>();
    r.value_type_size = sizeof(VT);
    r.index_type_size = sizeof(IT);

    r.was_matrix_sorted = mtx.is_sorted;

    compute_code_balances(k_entry.format, k_entry.is_gpu_kernel, false, r);

    x_out = std::move(x_csr);
    y_out = std::move(y_csr);

    return r;
}


template <typename VT, typename IT>
static BenchmarkResult
bench_spmv_ell(
                const Config & config,
                const MtxData<VT, IT> & mtx,
                const Kernel::entry_t & k_entry,
                DefaultValues<VT, IT> & defaults,
                std::vector<VT> &x_out,
                std::vector<VT> &y_out,

                const std::vector<VT> * x_in = nullptr)
{
    BenchmarkResult r;

    const ST nnz    = mtx.nnz;
    const ST n_rows = mtx.n_rows;
    // const ST n_cols = mtx.n_cols;

    ST n_els_per_row = config.n_els_per_row;

    V<VT, IT> values;
    V<IT, IT> col_idxs;

    log("converting to ell format start\n");

    bool col_major = k_entry.format == MatrixFormat::EllCm;

    if (!convert_to_ell<VT, IT>(mtx, col_major, n_els_per_row, col_idxs, values)) {
        r.is_result_valid = false;
        return r;
    }

    if (   n_els_per_row * n_rows != col_idxs.n_rows
        || n_els_per_row * n_rows != values.n_rows) {
        fprintf(stderr, "ERROR: converting matrix to ell format failed.\n");
        r.is_result_valid = false;
        return r;
    }

    log("converting to ell format end\n");

    V<VT, IT> x_ell = V<VT, IT>(mtx.n_cols);
    init_with_ptr_or_value(x_ell, x_ell.n_rows, x_in,
                           defaults.x, config.random_init_x);

    V<VT, IT> y_ell = V<VT, IT>(mtx.n_rows);
    std::uninitialized_fill_n(y_ell.data(), y_ell.n_rows, defaults.y);

    Kernel::fn_ell_t<VT, IT> kernel = k_entry.as_ell_kernel<VT, IT>();

    if (k_entry.is_gpu_kernel) {
#ifdef __NVCC__
        log("init GPU matrices start\n");
        VG<VT, IT> values_gpu(values);
        VG<IT, IT> col_idxs_gpu(col_idxs);
        VG<VT, IT> x_gpu(x_ell);
        VG<VT, IT> y_gpu(y_ell);
        log("init GPU matrices end\n");

        r = spmv<VT, IT>([&]() {
                const int num_blocks = (n_rows + default_block_size - 1) \
                                        / default_block_size;

                kernel<<<num_blocks, default_block_size>>>(n_rows,
                       n_els_per_row, col_idxs_gpu.data(), values_gpu.data(),
                       x_gpu.data(), y_gpu.data());
                },
                /* is_gpu_kernel */ true,
                config);

        y_ell = y_gpu.copy_from_device();
#endif
    }
    else {
        r = spmv<VT, IT>([&]() {
                kernel(n_rows, n_els_per_row,
                       col_idxs.data(), values.data(),
                       x_ell.data(), y_ell.data());
                },
                /* is_gpu_kernel */ false,
                config);
    }

    if (config.verify_result) {
        V<VT, IT> y_ref(y_ell.n_rows);
        std::uninitialized_fill_n(y_ref.data(), y_ref.n_rows, defaults.y);

        spmv_ell_reference(col_major,
                        n_rows, n_els_per_row,
                        col_idxs.data(), values.data(),
                        x_ell.data(), y_ref.data());

        r.is_result_valid &= spmv_verify(y_ref.data(), y_ell.data(),
                                         y_ell.n_rows, config.verbose_verification);
    }

    double mem_matrix_b =
              (double)sizeof(VT) * values.n_rows     // values
            + (double)sizeof(IT) * col_idxs.n_rows;  // col idxs

    double mem_x_b = (double)sizeof(VT) * x_ell.n_rows;
    double mem_y_b = (double)sizeof(VT) * y_ell.n_rows;

    double mem_b   = mem_matrix_b + mem_x_b + mem_y_b;

    r.mem_mb   = mem_b / 1e6;
    r.mem_m_mb = mem_matrix_b / 1e6;
    r.mem_x_mb  = mem_x_b / 1e6;
    r.mem_y_mb  = mem_y_b / 1e6;

    r.n_rows = mtx.n_rows;
    r.n_cols = mtx.n_cols;
    r.nnz    = nnz;

    r.duration_kernel_s = r.duration_total_s / r.n_calls;
    r.perf_gflops       = (double)nnz * 2.0
                          / r.duration_kernel_s
                          / 1e9; // Only count usefull flops

    r.value_type_str = type_name_from_type<VT>();
    r.index_type_str = type_name_from_type<IT>();
    r.value_type_size = sizeof(VT);
    r.index_type_size = sizeof(IT);

    r.was_matrix_sorted = mtx.is_sorted;

    r.fill_in_percent = ((double)(n_els_per_row * n_rows) / nnz - 1.0) * 100.0;
    r.nzr             = n_els_per_row;

    compute_code_balances(k_entry.format, k_entry.is_gpu_kernel, false, r);

    x_out = std::move(x_ell);
    y_out = std::move(y_ell);

    return r;
}


template <typename VT, typename IT>
static BenchmarkResult
bench_spmv_scs(
                const Config & config,
                const MtxData<VT, IT> & mtx,

                const Kernel::entry_t & k_entry,

                DefaultValues<VT, IT> & defaults,
                std::vector<VT> &x_out,
                std::vector<VT> &y_out,

                const std::vector<VT> * x_in = nullptr)
{
    log("allocate and place CPU matrices start\n");

    BenchmarkResult r;

    ScsData<VT, IT> scs;

    log("allocate and place CPU matrices end\n");
    log("converting to scs format start\n");

    if (!convert_to_scs<VT, IT>(mtx, config.chunk_size, config.sigma, scs)) {
        r.is_result_valid = false;
        return r;
    }

    log("converting to scs format end\n");

    V<VT, IT> x_scs(scs.n_cols);
    init_with_ptr_or_value(x_scs, x_scs.n_rows, x_in,
                           defaults.x, config.random_init_x);

    V<VT, IT> y_scs = V<VT, IT>(scs.n_rows_padded);
    std::uninitialized_fill_n(y_scs.data(),y_scs.n_rows, defaults.y);

    Kernel::fn_scs_t<VT, IT> kernel = k_entry.as_scs_kernel<VT, IT>();

    // std::cout << "scs: C: " << scs.C << " sigma: " << scs.sigma << "\n";
    // std::cout << "scs: n_rows: " << scs.n_rows << " n_rows_padded: " << scs.n_rows_padded << "\n";
    // std::cout << "scs: n_elements: " << scs.n_elements << "\n";
    //
    // print_vector("x", x_scs);
    // print_vector("y(pre)", y_scs);
    // print_vector("chunk_ptrs", scs.chunk_ptrs);
    // print_vector("chunk_lengths", scs.chunk_lengths);
    // print_vector("col_idxs", scs.col_idxs);
    // print_vector("values", scs.values);

    if (k_entry.is_gpu_kernel) {
#ifdef __NVCC__
        log("init GPU matrices start\n");
        VG<VT, IT> values_gpu(scs.values);
        VG<IT, IT> col_idxs_gpu(scs.col_idxs);
        VG<IT, IT> chunk_ptrs_gpu(scs.chunk_ptrs);
        VG<IT, IT> chunk_lengths_gpu(scs.chunk_lengths);
        VG<VT, IT> x_gpu(x_scs);
        VG<VT, IT> y_gpu(y_scs);
        log("init GPU matrices end\n");

        r = spmv<VT, IT>([&]() {
                const int num_blocks = (scs.n_rows_padded + default_block_size - 1) \
                                        / default_block_size;

                kernel<<<num_blocks, default_block_size>>>(
                        scs.C,
                        scs.n_chunks,
                        chunk_ptrs_gpu.data(), chunk_lengths_gpu.data(),
                        col_idxs_gpu.data(),
                        values_gpu.data(),
                        x_gpu.data(), y_gpu.data());
                },
                /* is_gpu_kernel */ true,
                config);

        y_scs = y_gpu.copy_from_device();
#endif
    }
    else {
        r = spmv<VT, IT>([&]() {
                kernel(scs.C,
                       scs.n_chunks,
                       scs.chunk_ptrs.data(), scs.chunk_lengths.data(),
                       scs.col_idxs.data(), scs.values.data(),
                       x_scs.data(), y_scs.data());
                },
                /* is_gpu_kernel */ false,
                config);
    }

    // print_vector("y", y_scs);

    if (config.verify_result) {
        V<VT, IT> y_ref(y_scs.n_rows);
        std::uninitialized_fill_n(y_ref.data(), y_ref.n_rows, defaults.y);

        spmv_scs_reference(scs.C,
                           scs.n_chunks,
                           scs.chunk_ptrs.data(), scs.chunk_lengths.data(),
                           scs.col_idxs.data(), scs.values.data(),
                           x_scs.data(), y_ref.data());

        r.is_result_valid &= spmv_verify(y_scs.data(), y_ref.data(),
                                         y_scs.n_rows, config.verbose_verification);
    }

    double mem_matrix_b =
              (double)sizeof(VT) * scs.n_elements     // values
            + (double)sizeof(IT) * scs.n_chunks       // chunk_ptrs
            + (double)sizeof(IT) * scs.n_chunks       // chunk_lengths
            + (double)sizeof(IT) * scs.n_elements;    // col_idxs

    double mem_x_b = (double)sizeof(VT) * scs.n_cols;
    double mem_y_b = (double)sizeof(VT) * scs.n_rows_padded;
    double mem_b   = mem_matrix_b + mem_x_b + mem_y_b;

    r.mem_mb   = mem_b / 1e6;
    r.mem_m_mb = mem_matrix_b / 1e6;
    r.mem_x_mb  = mem_x_b / 1e6;
    r.mem_y_mb  = mem_y_b / 1e6;

    r.n_rows = mtx.n_rows;
    r.n_cols = mtx.n_cols;
    r.nnz    = scs.nnz;

    r.duration_kernel_s = r.duration_total_s/ r.n_calls;
    r.perf_gflops       = (double)scs.nnz * 2.0
                          / r.duration_kernel_s
                          / 1e9;                   // Only count usefull flops

    r.value_type_str = type_name_from_type<VT>();
    r.index_type_str = type_name_from_type<IT>();
    r.value_type_size = sizeof(VT);
    r.index_type_size = sizeof(IT);

    r.was_matrix_sorted = mtx.is_sorted;

    r.fill_in_percent = ((double)scs.n_elements / scs.nnz - 1.0) * 100.0;
    r.C               = scs.C;
    r.sigma           = scs.sigma;

    compute_code_balances(k_entry.format, k_entry.is_gpu_kernel, false, r);

    x_out = std::move(x_scs);

    y_out.resize(scs.n_rows);
    for (int i = 0; i < scs.old_to_new_idx.n_rows; ++i) {
        y_out[i] = y_scs[scs.old_to_new_idx[i]];
    }

    return r;
}



/**
 * Benchmark the OpenMP/GPU spmv kernel with a matrix of dimensions \p n_rows x
 * \p n_cols.  If \p mmr is not NULL, then the matrix is read via the provided
 * MatrixMarketReader from file.
 */
template <typename VT, typename IT>
static BenchmarkResult
bench_spmv(const std::string & kernel_name,
           const Config & config,
           const Kernel::entry_t & k_entry,
           const MtxData<VT, IT> & mtx,
           DefaultValues<VT, IT> * defaults = nullptr,
           const std::vector<VT> * x_in = nullptr,
           std::vector<VT> * y_out_opt = nullptr
)
{
    BenchmarkResult r;

    std::vector<VT> y_out;
    std::vector<VT> x_out;

    DefaultValues<VT, IT> default_values;

    if (!defaults) {
        defaults = &default_values;
    }

    switch (k_entry.format) {
    case MatrixFormat::Csr:
        r = bench_spmv_csr<VT, IT>(config,
                                   mtx,
                                   k_entry, *defaults,
                                   x_out, y_out, x_in);
        break;
    case MatrixFormat::EllRm:
    case MatrixFormat::EllCm:
        r = bench_spmv_ell<VT, IT>(config,
                                   mtx,
                                   k_entry, *defaults,
                                   x_out, y_out, x_in);
        break;
    case MatrixFormat::SellCSigma:
        r = bench_spmv_scs<VT, IT>(config,
                                   mtx,
                                   k_entry, *defaults,
                                   x_out, y_out, x_in);
        break;    default:
        fprintf(stderr, "ERROR: SpMV format for kernel %s is not implemented.\n", kernel_name.c_str());
        return r;
    }


    if (config.verify_result_with_coo) {
        log("verify begin\n");

        bool ok = spmv_verify(kernel_name, mtx, x_out, y_out);

        r.is_result_valid = ok;

        log("verify end\n");
    }

    if (y_out_opt) *y_out_opt = std::move(y_out);

    // if (print_matrices) {
    //     printf("Matrices for kernel: %s\n", kernel_name.c_str());
    //     printf("A, is_col_major: %d\n", A.is_col_major);
    //     print(A);
    //     printf("b\n");
    //     print(b);
    //     printf("x\n");
    //     print(x);
    // }

    return r;
}

/**
 * @brief Return the file base name without an extension, empty if base name
 *        cannot be extracted.
 * @param file_name The file name to extract the base name from.
 * @return the base name of the file or an empty string if it cannot be
 *         extracted.
 */
static std::string
file_base_name(const char * file_name)
{
    if (file_name == nullptr) {
        return std::string{};
    }

    std::string file_path(file_name);
    std::string file;

    size_t pos_slash = file_path.rfind('/');

    if (pos_slash == file_path.npos) {
        file = std::move(file_path);
    }
    else {
        file = file_path.substr(pos_slash + 1);
    }

    size_t pos_dot = file.rfind('.');

    if (pos_dot == file.npos) {
        return file;
    }
    else {
        return file.substr(0, pos_dot);
    }
}

static void
print_histogram(const char * name, const Histogram & hist)
{
    size_t n_buckets = hist.bucket_counts().size();

    for (size_t i = 0; i < n_buckets; ++i) {
        printf("%-20s %3lu  %7ld  %7lu\n",
               name,
               i,
               hist.bucket_upper_bounds()[i],
               hist.bucket_counts()[i]);
    }
}

template <typename T>
static std::string
to_string(const Statistics<T> & stats)
{
    std::stringstream stream;

    stream << "avg: "     << std::scientific << std::setprecision(2) << stats.avg
           << " s: "      << std::scientific << std::setprecision(2) << stats.std_dev
           << " cv: "     << std::scientific << std::setprecision(2) << stats.cv
           << " min: "    << std::scientific << std::setprecision(2) << stats.min
           << " median: " << std::scientific << std::setprecision(2) << stats.median
           << " max: "    << std::scientific << std::setprecision(2) << stats.max;

    return stream.str();
}

template <typename T>
static void
print_matrix_statistics(
        const MatrixStats<T> & matrix_stats,
        const std::string & matrix_name)
{

    printf("##mstats %19s  %7s %7s  %9s  %5s %5s  %8s  %8s  %9s  %6s %6s\n",
           "name", "rows", "cols", "nnz", "nzr", "nzc", "maxrow",
           "density", "bandwidth",
           "sym", "sorted");

    const char * name = "unknown";

    if (!matrix_name.empty()) {
        name = matrix_name.c_str();
    }

    printf("#mstats %-20s  %7ld %7ld  %9ld  %5.2f %5.2f  %8.2e  %8.2e  %9lu  %6d %6d\n",
           name,
           matrix_stats.n_rows, matrix_stats.n_cols,
           matrix_stats.nnz,
           matrix_stats.row_lengths.avg,
           matrix_stats.col_lengths.avg,
           matrix_stats.row_lengths.max,
           matrix_stats.densitiy,
           (uint64_t)matrix_stats.bandwidths.max,
           matrix_stats.is_symmetric,
           matrix_stats.is_sorted);

    printf("#mstats-nzr %-20s %s\n", name, to_string(matrix_stats.row_lengths).c_str());
    printf("#mstats-nzc %-20s %s\n", name, to_string(matrix_stats.col_lengths).c_str());
    printf("#mstats-bw  %-20s %s\n", name, to_string(matrix_stats.bandwidths).c_str());

    print_histogram("#mstats-rows", matrix_stats.row_lengths.hist);
    print_histogram("#mstats-cols", matrix_stats.col_lengths.hist);
    print_histogram("#mstats-bws",  matrix_stats.bandwidths.hist);
}


static void
print_results(bool print_as_list,
              const std::string & name,
              const MatrixStats<double> & matrix_stats,
              const BenchmarkResult & r,
              int n_threads,
              int print_details
              )
{

    auto & rls = matrix_stats.row_lengths;
    auto & cls = matrix_stats.col_lengths;
    auto & bws  = matrix_stats.bandwidths;

    if (print_as_list) {
        printf("%2s %5s %7ld %7ld %9ld  %9.3e  %9.3e  %7.2e %7.2e  %8lu %-5s %-10s %3d "
               "%6.1f %3ld %2ld %2ld  "
               "%8.2e",
            r.value_type_str.c_str(),
            r.index_type_str.c_str(),
            (long)r.n_rows, (long)r.n_cols, (long)r.nnz,
            r.perf_gflops, r.mem_mb, r.duration_total_s,
            r.duration_kernel_s, r.n_calls,
            r.is_result_valid ? "OK" : "ERROR",
            name.c_str(),
            n_threads,
            r.fill_in_percent, (long)r.nzr, (long)r.C, (long)r.sigma,
            r.beta);

        if (print_details) {


            printf("  %1lu %1lu  %f %f  "
                   "%8.2e  "
                   "%8.2e %8.2e %8.2e %8.2e %8.2e %8.2e  "
                   "%8.2e %8.2e %8.2e %8.2e %8.2e %8.2e  "
                   "%8.2e %8.2e %8.2e",
                   r.value_type_size, r.index_type_size,
                   r.cb_a_0, r.cb_a_nzc,
                   matrix_stats.densitiy,
                   rls.avg, rls.std_dev, rls.cv, rls.min, rls.median, rls.max,
                   cls.avg, cls.std_dev, cls.cv, cls.min, cls.median, cls.max,
                   r.mem_m_mb, r.mem_x_mb, r.mem_y_mb);
        }
        printf("\n");

    }
    else {
        printf("matrix\n");
        printf("  dimensions     :  %ld x %ld\n", (long)r.n_rows, (long)r.n_cols);
        printf("  nnz            :  %lu\n", matrix_stats.nnz);
        printf("  density        :  %e\n", matrix_stats.densitiy);
        printf("  symmetric      :  %s (%d)\n",
               matrix_stats.is_symmetric ? "yes" : "no", matrix_stats.is_symmetric);
        printf("  sorted         :  %s (%d)\n",
               matrix_stats.is_sorted ? "yes" : "no", matrix_stats.is_sorted);
        printf("  nzr            :  avg: %8.2e s: %8.2e cv: %8.2e min: %8.2e median: %8.2e max %8.2e\n",
               rls.avg, rls.std_dev, rls.cv, rls.min, rls.median, rls.max);
        printf("  nzc            :  avg: %8.2e s: %8.2e cv: %8.2e min: %8.2e median: %8.2e max %8.2e\n",
               cls.avg, cls.std_dev, cls.cv, cls.min, cls.median, cls.max);
        printf("  bw             :  avg: %8.2e s: %8.2e cv: %8.2e min: %8.2e median: %8.2e max %8.2e\n",
               bws.avg, bws.std_dev, bws.cv, bws.min, bws.median, bws.max);

        printf("\n");
        printf("kernel\n");
        printf("name             :  %s\n", name.c_str());
        printf("data type        :  %-5s  size: %lu b\n",
               r.value_type_str.c_str(), r.value_type_size);
        printf("index type       :  %-5s  size: %lu b\n",
               r.index_type_str.c_str(), r.index_type_size);
        printf("nnz              :  %ld\n", (long)r.nnz);
        printf("performance      :  %9.3e GFLOP/s\n", r.perf_gflops);
        printf("memory           :  %9.3e MB  M: %9.3e MB  x: %9.3e MB  y: %9.3eMB\n",
               r.mem_mb, r.mem_m_mb, r.mem_x_mb, r.mem_y_mb);
        printf("duration total   :  %9.3e s\n", r.duration_total_s);
        printf("duration kernel  :  %9.3e s\n", r.duration_kernel_s);
        printf("repetitions      :  %lu\n", r.n_calls);
        printf("no. of threads   :  %d\n", n_threads);
        printf("fill-in          :  %.1f %%   beta: %f\n",
               r.fill_in_percent, r.beta);
        printf("nzr              :  %ld\n", r.nzr);
        printf("C                :  %ld\n", r.C);
        printf("sigma            :  %ld\n", r.sigma);
        printf("matrix sorted    :  %s\n", r.was_matrix_sorted ? "true" : "false");
        printf("solution         :  %s\n", r.is_result_valid ? "OK" : "ERROR");
        printf("code balance     :  a=0: %f b/flop a=1/nzc: %f b/flop\n",
               r.cb_a_0, r.cb_a_nzc);
        printf("\n");
    }
}

#include "test.cpp"

static void
usage()
{
    fprintf(stderr, "Usage:\n");
    fprintf(stderr,
            "  spmv-<omp|gpu> <martix-market-filename> [[kernel|all] [-sp|-dp|-all] [-c C] [-s Sigma] [-nzr NZR]]\n");
    fprintf(stderr,
            "  spmv-<omp|gpu> list\n");
    fprintf(stderr,
            "  spmv-<omp|gpu> test\n");
}


int main(int argc, char *argv[])
{
    Config config;
    bool print_list = false;
    int print_details = 0;
    int print_matrix_stats = 0;

    const char * file_name{};
    std::string kernel_to_benchmark { "csr" };
    std::string value_type = { "dp" };

    MARKER_INIT();

    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s [martix-market-filename] [matrix-format] "
                "[max-elems-per-row]\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }

    file_name = argv[1];

    if (!strcmp("test", file_name)) {
        test::run(Kernel::kernels());
        return 0;
    }
    else if (!strcmp("list", file_name)) {
        for (const auto & it : Kernel::kernels()) {
            const std::string & name = it.first;
            printf("%s\n", name.c_str());
        }
        return 0;
    }

    int args_start_index = 2;

    if (argc > 2) {
        std::string kernel_to_search = argv[2];

        auto & kernels = Kernel::kernels();

        bool kernel_exists = kernel_to_search == "all" ||
                    std::any_of(begin(kernels), end(kernels),
                        [&](const auto & it) {
                            return it.first == kernel_to_search;
                        });

        if (kernel_exists) {
            kernel_to_benchmark = kernel_to_search;
            args_start_index = 3;
        }
    }

    for (int i = args_start_index; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-nzr") {
            if (i + 1 < argc) {
                config.n_els_per_row = atol(argv[++i]);

                if (config.n_els_per_row < 1) {
                    fprintf(stderr, "ERROR: no. of elements per row must be >= 1.\n");
                    exit(EXIT_FAILURE);
                }
            }
        }
        else if (arg == "-c") {
            if (i + 1 < argc) {
                config.chunk_size = atol(argv[++i]);

                if (config.chunk_size < 1) {
                    fprintf(stderr, "ERROR: chunk size must be >= 1.\n");
                    exit(EXIT_FAILURE);
                }
            }
        }
        else if (arg == "-s") {
            if (i + 1 < argc) {
                config.sigma = atol(argv[++i]);

                if (config.sigma < 1) { // || config.sigma < config.chunk_size) {
                    fprintf(stderr, "ERROR: sigma must be >= 1.\n");
                    exit(EXIT_FAILURE);
                }
            }
        }
        else if (arg == "-rand-x") {
            config.random_init_x = true;
        }
        else if (arg == "-list") {
            print_list = true;
        }
        else if (arg == "-log") {
            g_log = true;
        }
        else if (arg == "-dp") {
            value_type = "dp";
        }
        else if (arg == "-sp") {
            value_type = "sp";
        }
        else if (arg == "-all") {
            value_type = "all";
        }
        else if (arg == "-sort") {
            config.sort_matrix = true;
        }
        else if (arg == "-no-sort") {
            config.sort_matrix = false;
        }
        else if (arg == "-verify") {
            config.verify_result = true;
        }
        else if (arg == "-no-verify") {
            config.verify_result = false;
        }
        else if (arg == "-v") {
            ++print_details;
        }
        else if (arg == "-print-matrix-stats") {
            print_matrix_stats = 1;
        }
        else {
            fprintf(stderr, "ERROR: unknown argument.\n");
            usage();
            exit(EXIT_FAILURE);
        }
    }

    // TODO: print current configuration

    {
        uint64_t n_rows{};
        uint64_t n_cols{};
        uint64_t nnz{};

        get_mtx_dimensions(file_name, n_rows, n_cols, nnz);

        if (n_rows <= 0 || n_cols <= 0 || nnz <= 0) {
            fprintf(stderr, "Number of rows/columns/nnz in mtx file must be > 0.\n");
            return 1;
        }

        if (!is_type_large_enough_for_mtx_sizes<ST>(file_name)) {
            fprintf(stderr, "ERROR: size type is not large enough for matrix dimensions/nonzeros.\n");
            return 1;
        }
    }


    if (print_matrix_stats) {
        std::string matrix_name = file_base_name(file_name);

        MtxData<double, int> mtx = read_mtx_data<double, int>(file_name, config.sort_matrix);
        MatrixStats<double> matrix_stats = get_matrix_stats(mtx);
        print_matrix_statistics(matrix_stats, matrix_name);
        return 0;
    }

    int n_cpu_threads = 1;
#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp single
        n_cpu_threads = omp_get_max_threads();
    }
#endif

#if 0
    BenchmarkResult res;
    res = bench_spmv<VT, IT>(fstream, n_rows, n_cols, nnz,
                                           &nelems_per_row, alignment, k);

    ST n_fill_in = k == KernelType::ell ? n_rows * nelems_per_row - nnz : 0;
    double fill_in_fraction = ((double)n_fill_in / (double)nnz) * 100.0;


     std::printf(
         "%5ld %5ld sp: %9.3e GFLOP/s  (%9.3e MB)  "
         "kernel: %s, cpu-threads: %d, reps: %ld  dur: %e s\n",
         (long)n_rows, (long)n_cols, res.perf_gflops, res.memory_used_mb, k_type.c_str(),
         n_cpu_threads, res.num_spmv_calls, res.duration_s
                 );

     std::printf(
         "%8ld %8ld  nnz: %10ld  fill-in: %10ld  %.3f %%  max_el_per_row: %4ld\n",
         (long)n_rows, (long)n_cols,
         (long)nnz, (long)n_fill_in, fill_in_fraction, (long)nelems_per_row
                 );

    std::fflush(stdout);
#endif
#ifdef BENCHMARK_COMPLEX
    std::printf("#rows cols   sp: %7s %7s  dp: %7s %7s   csp: %7s %7s  cdp: %7s %7s  "
                "kernel          layout #threads\n",
                "GFLOP/s", "MB",
                "GFLOP/s", "MB",
                "GFLOP/s", "MB",
                "GFLOP/s", "MB");
#else
    if (print_list) {
        printf("#%s %-4s %-7s %-7s %-9s  "
               "%-10s %-9s  %-7s  %-7s  %-8s  %-5s %-10s "
               "%-3s %-6s %-3s %-2s %-2s %-8s",
               "fp", "idxt", "rows", "cols", "nnz",
               "p(GFLOP/s)", "MB", "t (s)", "k (s)", "reps", "ver", "name",
               "#th", "fi(%)", "nzr", "C", "S", "beta");
        if (print_details) {
            printf(" %s %s  %s %s  "
                   "%-8s  "
                   " %-8s %-8s %-8s %-8s %-8s %-8s  "
                   " %-8s %-8s %-8s %-8s %-8s %-8s  "
                   " %-8s %-8s %-8s",
                   "svt", "sit", "B(a=0)", "B(a=1/nzc)",
                   "dens.",
                   "rows-avg", "sd", "cv", "min", "mean", "max",
                   "cols-avg", "sd", "cv", "min", "mean", "max",
                   "memM(MB)", "memX(MB)", "memY(MB)");
        }
        printf("\n");
    }
#endif

    // long file_pos = ftell(f);

    MatrixStats<double> matrix_stats;
    bool matrix_stats_computed = false;

    for (auto & it : Kernel::kernels()) {

        const std::string & name = it.first;

        if (name != kernel_to_benchmark && kernel_to_benchmark != "all") {
            continue;
        }

        for (auto & it2 : it.second) {
            std::type_index k_float_type = std::get<0>(it2.first);
            std::type_index k_index_type = std::get<1>(it2.first);

            Kernel::entry_t & k_entry = it2.second;

            BenchmarkResult result;
            bool result_valid = true;

            log("benchmarking kernel: %s\n", name.c_str());

            if (k_float_type == std::type_index(typeid(float)) &&
                k_index_type == std::type_index(typeid(int))   &&
                (value_type == "sp" || value_type == "all")) {
                if (!is_type_large_enough_for_mtx_sizes<int>(file_name)) {
                    fprintf(stderr, "ERROR: matrix dimensions/nnz exceed size of index type int\n");
                    continue;
                }
                MtxData<float, int> mtx = read_mtx_data<float, int>(file_name, config.sort_matrix);
                if (!matrix_stats_computed) {
                    matrix_stats = get_matrix_stats(mtx);
                    matrix_stats_computed = true;
                }
                result = bench_spmv<float, int>(name, config, k_entry, mtx);
            }
            else if (k_float_type == std::type_index(typeid(double)) &&
                     k_index_type == std::type_index(typeid(int))    &&
                (value_type == "dp" || value_type == "all")) {
                if (!is_type_large_enough_for_mtx_sizes<int>(file_name)) {
                    fprintf(stderr, "ERROR: matrix dimensions/nnz exceed size of index type int\n");
                    continue;
                }
                MtxData<double, int> mtx = read_mtx_data<double, int>(file_name, config.sort_matrix);
                if (!matrix_stats_computed) {
                    matrix_stats = get_matrix_stats(mtx);
                    matrix_stats_computed = true;
                }
                result = bench_spmv<double, int>(name, config, k_entry, mtx);
            }
#ifdef BENCHMARK_COMPLEX
            else if (k_float_type == std::type_index(typeid(std::complex<float>))) {
                results[2] = bench_gemv<std::complex<float>>(name, n_rows, n_cols, k_entry, file_name);
            }
            else if (k_float_type == std::type_index(typeid(std::complex<double>))) {
                results[3] = bench_gemv<std::complex<double>>(name, n_rows, n_cols, k_entry, file_name);
            }
#endif
            else {
                result_valid = false;
            }

            log("benchmarking kernel: %s end\n", name.c_str());

            if (result_valid) {
                print_results(print_list, name, matrix_stats, result, n_cpu_threads, print_details);
            }
        }
    }

    log("main end\n");

    MARKER_DEINIT();

    return 0;
}

