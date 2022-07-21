#include <iomanip>
#include <random>
#include <unordered_set>

namespace test {

long g_tests_failed = 0;
long g_tests_total  = 0;

static void test_failed()    { ++g_tests_failed; ++g_tests_total; }
static void test_succeeded() { ++g_tests_total; }

static void
tlog(const char * format, ...)
{
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);

    fflush(stdout);
}

static void
terror(const char * format, ...)
{
    va_list args;
    char buffer[1024];
    snprintf(buffer, sizeof(buffer), "\n%s", format);

    va_start(args, format);
    vprintf(buffer, args);
    va_end(args);

    fflush(stdout);
}

#define TO_STRING_(a)   #a
#define TO_STRING(a)    TO_STRING_(a)

#define tassert(a)   _tassert_impl((a), TO_STRING(a), __FILE__, __LINE__)

static void
_tassert_impl(bool equal, const char * expression, const char * file, int line)
{
    if (!equal) {
        test_failed();
        terror("ERROR: assert failed: %s:%d  expression: %s\n",
               file, line, expression);
        exit(EXIT_FAILURE);
    }
    else {
        test_succeeded();
    }
}


template <typename T>
static void
test_will_add_overflow_signed_t()
{
    tassert(will_add_overflow(T{1}, std::numeric_limits<T>::max()));
    tassert(will_add_overflow(std::numeric_limits<T>::max(), T{1}));

    tassert(!will_add_overflow(T{1}, std::numeric_limits<T>::max() - T{1}));
    tassert(!will_add_overflow(std::numeric_limits<T>::max() - T{1}, T{1}));

    tassert(will_add_overflow(T{-1}, std::numeric_limits<T>::min()));
    tassert(will_add_overflow(std::numeric_limits<T>::min(), T{-1}));

    tassert(!will_add_overflow(T{-1}, std::numeric_limits<T>::min() + T{1}));
    tassert(!will_add_overflow(std::numeric_limits<T>::min() + T{1}, T{-1}));

    tassert(!will_add_overflow(std::numeric_limits<T>::max(),
                               std::numeric_limits<T>::min()));

}

static void
test_will_add_overflow()
{
    test_will_add_overflow_signed_t<int64_t>();
    test_will_add_overflow_signed_t<int32_t>();
    test_will_add_overflow_signed_t<long>();
    test_will_add_overflow_signed_t<int>();
}

template <typename T>
static void
test_will_mult_overflow_signed_t()
{
    tassert(will_mult_overflow(std::numeric_limits<T>::min(), std::numeric_limits<T>::max()));

    tassert(!will_mult_overflow(T{1}, std::numeric_limits<T>::max()));
    tassert(!will_mult_overflow(T{1}, std::numeric_limits<T>::min()));

    // only true for two's complement
    tassert(will_mult_overflow(T{-1}, std::numeric_limits<T>::min()));

    tassert(!will_mult_overflow(T{-1}, std::numeric_limits<T>::min() + 1));

    tassert(will_mult_overflow(T{ 4}, std::numeric_limits<T>::max() / 2));
    tassert(will_mult_overflow(T{-4}, std::numeric_limits<T>::max() / 2));

    tassert(will_mult_overflow(T{ 4}, std::numeric_limits<T>::min() / 2));
    tassert(will_mult_overflow(T{-4}, std::numeric_limits<T>::min() / 2));
}

static void
test_will_mult_overflow()
{
    test_will_mult_overflow_signed_t<int64_t>();
    test_will_mult_overflow_signed_t<int32_t>();
    test_will_mult_overflow_signed_t<long>();
    test_will_mult_overflow_signed_t<int>();
}

template <typename VT, typename IT>
struct System
{
    ST n_rows{};
    ST n_cols{};

    MtxData<VT, IT> mtx;

    std::vector<VT> x_ref;
    std::vector<VT> y_ref;
};

template <typename VT, typename IT>
static void
mtx_to_stream(
        const MtxData<VT, IT> & mtx,
        std::istringstream & is)
{
    std::stringstream s;

    for (ST i = 0; i < mtx.nnz; ++i) {
        s << mtx.I[i] + 1 << " " << mtx.J[i] + 1
          << " "
          << std::setprecision(std::numeric_limits<VT>::digits10)
          << std::scientific
          << mtx.values[i] << "\n";
    }

    is.str(s.str());
}


template <typename VT, typename IT>
static System<VT, IT>
generate_diagonal_system(ST N)
{
    System<VT, IT> m;
    MtxData<VT, IT> & mtx = m.mtx;

    m.n_rows = N;
    m.n_cols = N;

    mtx.n_rows = N;
    mtx.n_cols = N;
    mtx.nnz = N;

    mtx.I.reserve(mtx.n_rows);
    mtx.J.reserve(mtx.n_rows);
    mtx.values.reserve(mtx.n_rows);

    for (ST i = 0; i < mtx.n_rows; ++i) {
        mtx.I.push_back(i);
        mtx.J.push_back(i);
        mtx.values.push_back(VT(i + 1));
    }

    m.x_ref.resize(m.n_cols);
    std::iota(std::begin(m.x_ref), std::end(m.x_ref), VT{1.0});

    m.y_ref.assign(m.n_rows, VT{});

    for (ST i = 0; i < m.mtx.nnz; ++i) {
        m.y_ref[mtx.I[i]] += mtx.values[i] * m.x_ref[mtx.J[i]];
    }

    return m;
}


template <typename VT, typename IT>
static System<VT, IT>
generate_random_system(ST N)
{
    System<VT, IT> m;
    MtxData<VT, IT> & mtx = m.mtx;

    m.n_rows = N;
    m.n_cols = N;

    mtx.n_rows = N;
    mtx.n_cols = N;
    mtx.nnz = 0;

    std::random_device rnd;
    std::mt19937 generator(rnd());

    ST max_nzr = m.n_cols > 10 ? m.n_cols / 10 : 1;

    std::uniform_int_distribution<ST> rand_nzr(1, max_nzr);
    std::uniform_int_distribution<IT> rand_col(0, m.n_cols - 1);

    // Generate values for x and values as ST.
    std::uniform_int_distribution<ST> rand_value(-m.n_rows, m.n_rows);

    ST max_nnz = mtx.n_rows * max_nzr;
    mtx.I.reserve(max_nnz);
    mtx.J.reserve(max_nnz);
    mtx.values.reserve(max_nnz);

    for (ST row = 0; row < m.n_rows; ++row) {
        ST nzr = rand_nzr(generator);
        if (nzr > m.n_cols) {
            fprintf(stderr, "ERROR: cannot create more non-zeros per row as columns exist.\n");
            exit(EXIT_FAILURE);
        }

        std::unordered_set<IT> non_zero_cols;

        while (non_zero_cols.size() < (size_t)nzr) {
            IT col = rand_col(generator);
            non_zero_cols.insert(col);
        }

        for (ST col : non_zero_cols) {
            mtx.I.push_back(row);
            mtx.J.push_back(col);
            mtx.values.push_back(VT(rand_value(generator)));
        }

        mtx.nnz += nzr;
    }

    m.x_ref.resize(m.n_cols);
    std::iota(std::begin(m.x_ref), std::end(m.x_ref), VT{1.0});

    m.y_ref.assign(m.n_rows, VT{});

    for (ST i = 0; i < m.mtx.nnz; ++i) {
        m.y_ref[mtx.I[i]] += mtx.values[i] * m.x_ref[mtx.J[i]];
    }

    return m;
}


template <typename VT, typename IT>
static void
test_convert_to_scs(
        const std::string & name,
        const Kernel::entry_t & k_entry,
        ST C = 4,
        ST sigma = -1)
{
    System<VT, IT> m{generate_random_system<VT, IT>(100)};
    std::istringstream s;
    MtxData<VT, IT> & mtx = m.mtx;
    ScsData<VT, IT> scs;

    mtx_to_stream(mtx, s);

    convert_to_scs<VT, IT>(mtx, C, sigma, scs);

    // TODO: extend tests...
    tassert(C == scs.C);
    if (sigma < 1) {
        tassert(1 == scs.sigma);
    }
    else {
        tassert(sigma == scs.sigma);
    }

}

template <typename VT, typename IT>
static void
test_mtx_is_symmetric()
{
    tlog("testing is_mtx_symmetric\n");

    {
        System<VT, IT> m{generate_diagonal_system<VT, IT>(10)};
        MtxData<VT, IT> & mtx = m.mtx;

        tassert(is_mtx_symmetric(mtx));
    }

    for (int i = 1; i <= 10; ++i) {
        System<VT, IT> m{generate_diagonal_system<VT, IT>(10)};
        MtxData<VT, IT> & mtx = m.mtx;

        ++mtx.nnz;
        mtx.I.push_back(i);
        if (i < 10) {
            mtx.J.push_back(i + 1);
        }
        else {
            mtx.J.push_back(1);
        }

        tassert(!is_mtx_symmetric(mtx));
    }

}

template <typename VT, typename IT>
static bool
test_kernel(const std::string & name,
            const Kernel::entry_t & k_entry,
            const System<VT, IT> & sys,
            const Config * default_config = nullptr)
{
    const MtxData<VT, IT> & mtx = sys.mtx;

    DefaultValues<VT, IT> defaults;

    defaults.x_values = const_cast<VT *>(sys.x_ref.data());
    defaults.n_x_values = sys.x_ref.size();

    defaults.x = 8.765432;
    defaults.y = 7.654321;

    // printf("converting system to mtx\n"); fflush(stdout);


    std::vector<VT> y_actual;

    Config config;

    if (default_config) {
        config = *default_config;
    }

    config.random_init_x  = false;
    config.random_init_A  = false;
    config.verify_result  = false;
    config.n_repetitions  = 1;


    // printf("calling bench_spmv\n"); fflush(stdout);
    BenchmarkResult r =
            bench_spmv<VT, IT>(
                name,
                config,
                k_entry,
                mtx,
                &defaults,
                &sys.x_ref,
                &y_actual);

    // print_vector("y(ref)", m.y_ref.data(), m.y_ref.data() + m.y_ref.size());

    if (y_actual.size() != sys.y_ref.size()) {
        terror("ERROR: size of y is incorrect.\n");
        return false;
    }

    std::vector<VT> y_ref = sys.y_ref;

    // y_ref contains the value of y = A x, so we have to adjust it for the
    // default y value and the no. of spmvs performed.
    for (size_t i = 0; i < sys.y_ref.size(); ++i) {
        y_ref[i] = y_ref[i] * r.n_total_calls + defaults.y;
    }

    VT max_rel_error_found;
    ST error_counter =
        compare_arrays(y_ref.data(),
                       y_actual.data(),
                       sys.n_rows,
                       false /* verbose */,
                       max_rel_error<VT>::value,
                       max_rel_error_found);

    if (error_counter > 0) {
        printf("e"); fflush(stdout);
        terror("ERROR: spmv kernel %s (fp size %ld, idx size %ld) is incorrect, "
               "relative error > %e for %ld/%ld elements. Max rel. error. found %e.\n",
               name.c_str(), sizeof(VT), sizeof(IT),
               max_rel_error<VT>::value,
               (long)error_counter, (long)sys.n_rows,
               max_rel_error_found);
        terror("kernel: %s(%ld, %ld)   ERROR\n",
               name.c_str(), sizeof(VT), sizeof(IT));
        test_failed();
    }
    else {
        printf("."); fflush(stdout);
        test_succeeded();
    }

    return error_counter == 0;
}


template <typename VT, typename IT>
static void
test_kernel_with_config(
        const std::string & name,
        const Kernel::entry_t & k_entry,
        Config * config = nullptr)
{
    if (k_entry.format == MatrixFormat::SellCSigma) {
        ST C = 4;
        ST sigma = -1;

        if (config) {
            C = config->chunk_size;
            sigma = config->sigma;
        }

        test_convert_to_scs<VT, IT>(name, k_entry, C, sigma);
    }

    for (ST i = 7; i <= 100000; i *= 10) {
        test_kernel<VT, IT>(name,
                            k_entry,
                            generate_diagonal_system<VT, IT>(i),
                            config);
    }

    std::random_device rnd;
    std::mt19937 gen(rnd());
    std::uniform_int_distribution<ST> dist(1, 1000);

    for (ST i = 0; i < 10; ++i) {
        ST system_size = dist(gen);

        test_kernel<VT, IT>(name,
                            k_entry,
                            generate_random_system<VT, IT>(system_size),
                            config);
    }

    return;
}

static void
test_kernels(const Kernel::kernels_t & kernels)
{
    for (auto & it : kernels) {
        const std::string & name = it.first;

        tlog("\ntesting kernel: %s\n", name.c_str());

        for (auto & it2 : it.second) {
            std::type_index k_float_type = std::get<0>(it2.first);
            std::type_index k_index_type = std::get<1>(it2.first);

            const Kernel::entry_t & k_entry = it2.second;

            Config * config{};
            Config default_config{};

            std::vector<std::tuple<ST, ST>> cfgs;

            if (k_entry.format == MatrixFormat::SellCSigma) {
                 cfgs = {
                    std::make_tuple( 4, -1 ), std::make_tuple( 4,    4 ), std::make_tuple( 4, 128 ),
                    std::make_tuple( 8, -1 ), std::make_tuple( 8,   32 ), std::make_tuple( 8, 128 ),
                    std::make_tuple(16, -1 ), std::make_tuple(16,   64 ), std::make_tuple(16, 128 ),
                    std::make_tuple(32, -1 ), std::make_tuple(32,  128 ), std::make_tuple(32, 128 ),
                    // {64, -1 }, {64,  256 }, {64, 128 }
                };
            }
            else {
                cfgs = {std::make_tuple( 0, 0 )};
            }

            for (const auto & pair : cfgs) {
                config = &default_config;
                config->chunk_size = std::get<0>(pair);
                config->sigma      = std::get<1>(pair);

                if (k_float_type == std::type_index(typeid(float)) &&
                    k_index_type == std::type_index(typeid(int))) {
                    test_kernel_with_config<float, int>(
                                name, k_entry, config);
                }
                else if (k_float_type == std::type_index(typeid(double)) &&
                         k_index_type == std::type_index(typeid(int))) {
                    test_kernel_with_config<double, int>(
                                name, k_entry,
                                config);
                }
            }
        }
    }
}

static MtxReader
get_mtx_reader_from_stream(std::istream & stream)
{
    MtxReader m;
    try {
        m = MtxReader{stream};
    }
    catch (const std::exception & e) {
        tassert(false);
        fprintf(stderr, "ERROR: creating MtxReader failed %s\n", e.what());
    }

    return m;
}

static void
test_mtx_reader()
{
    tlog("testing MtxReader\n");

    {
        uint64_t n_rows = 10, n_cols = 10, nnz = 12345;

        std::stringstream stream;
        stream << "%%MatrixMarket matrix coordinate real general\n"
               << n_rows << " " << n_cols << " " << nnz << "\n";
        MtxReader m = get_mtx_reader_from_stream(stream);

        tassert(m.is_data_type_present() == true);
        tassert(m.is_real() == true);
        tassert(m.is_integer() == false);
        tassert(m.is_complex() == false);
        tassert(m.is_general() == true);
        tassert(m.is_symmetric() == false);

        tassert(m.m() == n_rows);
        tassert(m.n() == n_cols);
        tassert(m.nnz() == nnz);
    }

    {
        uint64_t n_rows = 10, n_cols = 10, nnz = 12345;

        std::stringstream stream;
        stream << "%%MatrixMarket matrix coordinate real\n"
               << n_rows << " " << n_cols << " " << nnz << "\n";
        MtxReader m = get_mtx_reader_from_stream(stream);

        tassert(m.is_data_type_present() == true);
        tassert(m.is_real() == true);
        tassert(m.is_integer() == false);
        tassert(m.is_complex() == false);
        tassert(m.is_general() == true);
        tassert(m.is_symmetric() == false);

        tassert(m.m() == n_rows);
        tassert(m.n() == n_cols);
        tassert(m.nnz() == nnz);
    }

    {
        uint64_t n_rows = 10, n_cols = 10, nnz = 12345;

        std::stringstream stream;
        stream << "%%MatrixMarket matrix coordinate real symmetric\n"
               << n_rows << " " << n_cols << " " << nnz << "\n";
        MtxReader m = get_mtx_reader_from_stream(stream);

        tassert(m.is_data_type_present() == true);
        tassert(m.is_real() == true);
        tassert(m.is_integer() == false);
        tassert(m.is_complex() == false);
        tassert(m.is_general() == false);
        tassert(m.is_symmetric() == true);

        tassert(m.m() == n_rows);
        tassert(m.n() == n_cols);
        tassert(m.nnz() == nnz);
    }

    {
        uint64_t n_rows = 10, n_cols = 10, nnz = 12345;

        std::stringstream stream;
        stream << "%%MatrixMarket matrix coordinate integer\n"
               << n_rows << " " << n_cols << " " << nnz << "\n";
        MtxReader m = get_mtx_reader_from_stream(stream);

        tassert(m.is_data_type_present() == true);
        tassert(m.is_real() == false);
        tassert(m.is_integer() == true);
        tassert(m.is_complex() == false);
        tassert(m.is_general() == true);
        tassert(m.is_symmetric() == false);

        tassert(m.m() == n_rows);
        tassert(m.n() == n_cols);
        tassert(m.nnz() == nnz);
    }

    {
        uint64_t n_rows = 10, n_cols = 10, nnz = 12345;

        std::stringstream stream;
        stream << "%%MatrixMarket matrix coordinate complex\n"
               << n_rows << " " << n_cols << " " << nnz << "\n";
        MtxReader m = get_mtx_reader_from_stream(stream);

        tassert(m.is_data_type_present() == true);
        tassert(m.is_real() == false);
        tassert(m.is_integer() == false);
        tassert(m.is_complex() == true);
        tassert(m.is_general() == true);
        tassert(m.is_symmetric() == false);

        tassert(m.m() == n_rows);
        tassert(m.n() == n_cols);
        tassert(m.nnz() == nnz);
    }

    {
        uint64_t n_rows = 12, n_cols = 14, nnz = 54321;

        std::stringstream stream;
        stream << "%%MatrixMarket matrix coordinate real\n"
               << "% comment 1\n"
               << "% comment 2\n"
               << n_rows << " " << n_cols << " " << nnz << "\n";
        MtxReader m = get_mtx_reader_from_stream(stream);

        tassert(m.is_data_type_present() == true);
        tassert(m.is_real() == true);
        tassert(m.is_integer() == false);
        tassert(m.is_complex() == false);
        tassert(m.is_general() == true);
        tassert(m.is_symmetric() == false);

        tassert(m.m() == n_rows);
        tassert(m.n() == n_cols);
        tassert(m.nnz() == nnz);
    }

    {
        uint64_t n_rows = 12, n_cols = 14, nnz = 54321;

        std::stringstream stream;
        stream << "%%MatrixMarket xmatrix coordinate real\n"
               << "% comment 1\n"
               << "% comment 2\n"
               << n_rows << " " << n_cols << " " << nnz << "\n";

        bool throwed = false;

        try {
            MtxReader m{stream};
        }
        catch (const std::exception & e) {
            throwed = true;
        }

        tassert(throwed == true);
    }

    {
        uint64_t n_rows = 12, n_cols = 14, nnz = 54321;

        std::stringstream stream;
        stream << "%%MatrixMarket matrix xcoordinate real\n"
               << "% comment 1\n"
               << "% comment 2\n"
               << n_rows << " " << n_cols << " " << nnz << "\n";

        bool throwed = false;

        try {
            MtxReader m{stream};
        }
        catch (const std::exception & e) {
            throwed = true;
        }

        tassert(throwed == true);
    }

    {
        uint64_t n_rows = 12, n_cols = 14, nnz = 54321;

        std::stringstream stream;
        stream << "%%MatrixMarket matrix coordinate pattern\n"
               << "% comment 1\n"
               << "% comment 2\n"
               << n_rows << " " << n_cols << " " << nnz << "\n";

        bool throwed = false;

        try {
            MtxReader m{stream};
        }
        catch (const std::exception & e) {
            throwed = true;
        }

        tassert(throwed == true);
    }
}

static void
test_read_mtx_data()
{
    tlog("testing read_mtx_data\n");

    {
        uint64_t n_rows = 10, n_cols = 10, nnz = 10;

        std::stringstream header_stream;
        header_stream
               << "%%MatrixMarket matrix coordinate real symmetric\n"
               << n_rows << " " << n_cols << " " << nnz << "\n";

        std::stringstream data_stream;
        for (uint64_t row = 1; row <= n_rows; ++row) {
            data_stream << row << " " << 1 << " " << (double)row << "\n";
        }

        MtxReader m = get_mtx_reader_from_stream(header_stream);

        tassert(m.m() == n_rows);
        tassert(m.n() == n_cols);
        tassert(m.nnz() == nnz);

        MtxData<double, int> mtx = read_mtx_data<double, int>(
                    data_stream,
                    m.m(), m.n(), m.nnz(),
                    m.is_symmetric());

        tassert(mtx.n_rows == (ST)n_rows);
        tassert(mtx.n_cols == (ST)n_cols);
        tassert(mtx.nnz    == (ST)(n_rows + n_cols - 1));

        uint64_t i = 0;
        for (uint64_t row = 0; row < n_rows; ++row) {
              if (mtx.I[i] == (int)row) {
                  tassert(mtx.J[i] == 0);
                  tassert(mtx.values[i] == (double)(row + 1));
              }
              ++i;
        }

        for (uint64_t col = 1; col < n_cols; ++col) {
              if (mtx.J[i] == (int)col) {
                  tassert(mtx.I[i] == 0);
                  tassert(mtx.values[i] == (double)(col + 1));
              }
              ++i;
        }
    }

    {
        uint64_t n_rows = 10, n_cols = 10, nnz = 10;

        std::stringstream header_stream;
        header_stream
               << "%%MatrixMarket matrix coordinate real general\n"
               << n_rows << " " << n_cols << " " << nnz << "\n";

        std::stringstream data_stream;
        for (uint64_t row = 1; row <= n_rows; ++row) {
            data_stream << row << " " << 1 << " " << (double)row << "\n";
        }

        MtxReader m = get_mtx_reader_from_stream(header_stream);

        tassert(m.m()   == n_rows);
        tassert(m.n()   == n_cols);
        tassert(m.nnz() == nnz);

        MtxData<double, int> mtx = read_mtx_data<double, int>(
                    data_stream,
                    m.m(), m.n(), m.nnz(),
                    m.is_symmetric());

        tassert(mtx.n_rows == (ST)n_rows);
        tassert(mtx.n_cols == (ST)n_cols);
        tassert(mtx.nnz    == (ST)nnz);

        uint64_t i = 0;
        for (uint64_t row = 0; row < n_rows; ++row) {
              if (mtx.I[i] == (int)row) {
                  tassert(mtx.J[i] == 0);
                  tassert(mtx.values[i] == (double)(row + 1));
              }
              ++i;
        }
    }
}

#ifdef WITH_SAMPLE_HPCG

#ifndef SAMPLE_PREFIX_HPCG
#define SAMPLE_PREFIX_HPCG ""
#endif

static void
test_mtx_gen_hpcg()
{
    tlog("testing mtx gen:hpcg\n");

    using VT = double;
    using IT = int;
    using namespace mtx_gen_hpcg;

    const char * ref_hpcg_matrix_path = "test-data/hpcg-5-5-5.mtx";

    MtxData<VT, IT> ref = read_mtx_data<VT, IT>(ref_hpcg_matrix_path, true);

    MtxData<VT, IT> actual = generate_mtx<VT, IT>(SAMPLE_PREFIX_HPCG "5");

    tassert(ref.nnz    == actual.nnz);
    tassert(ref.n_rows == actual.n_rows);
    tassert(ref.n_cols == actual.n_cols);

    tassert(ref.is_sorted    == actual.is_sorted);
    tassert(ref.is_symmetric == actual.is_symmetric);

    tassert(is_mtx_symmetric_fast(actual));
    tassert(is_mtx_sorted(actual));

    for (ST i = 0; i < ref.nnz; ++i) {
        tassert(ref.I[i]      == actual.I[i]);
        tassert(ref.J[i]      == actual.J[i]);
        tassert(ref.values[i] == actual.values[i]);
    }

    tassert(is_sample_hpcg(SAMPLE_PREFIX_HPCG "5"));
    tassert(is_sample_hpcg(SAMPLE_PREFIX_HPCG "5:5"));
    tassert(is_sample_hpcg(SAMPLE_PREFIX_HPCG "5:5:5"));

    tassert(!is_sample_hpcg("gen:hpc:5"));
    tassert(!is_sample_hpcg("genn:hpcg:5"));

    auto test_dims = [](const char * desc, uint64_t ref_x, uint64_t ref_y, uint64_t ref_z, bool ref_ret) {
        uint64_t nx, ny, nz;

        tassert(get_dimensions(desc, nx, ny, nz) == ref_ret);

        if (!ref_ret) return;

        tassert(ref_x == nx);
        tassert(ref_y == ny);
        tassert(ref_z == nz);
    };

    //        |       input              |     expected      |
    //                                    nx   ny   nz   return value
    test_dims(SAMPLE_PREFIX_HPCG "5:6:7", 5ul, 6ul, 7ul, true);
    test_dims(SAMPLE_PREFIX_HPCG "5:6",   5ul, 6ul, 6ul, true);
    test_dims(SAMPLE_PREFIX_HPCG "5",     5ul, 5ul, 5ul, true);

    test_dims("5:6:7", 5ul, 6ul, 7ul, true);
    test_dims("5:6",   5ul, 6ul, 6ul, true);
    test_dims("5",     5ul, 5ul, 5ul, true);

    // This is something where we should throw an error, but as parsing is
    // currently implemented x will be 5, y = 7, and z = y;
    test_dims("5x6:7", 5ul, 7ul, 7ul, true);

    test_dims("x5x6:7", 5ul, 7ul, 7ul, false);

    test_dims("0:6:7",  5ul, 7ul, 7ul, false);
    test_dims("5:0:7",  5ul, 7ul, 7ul, false);
    test_dims("5:6:0",  5ul, 7ul, 7ul, false);

    test_dims("-5:6:1",  5ul, 7ul, 7ul, false);
    test_dims("5:-6:1",  5ul, 7ul, 7ul, false);
    test_dims("5:6:-1",  5ul, 7ul, 7ul, false);
}
#endif

static void
test_histogram()
{
    tlog("testing Histogram\n");

    {
        Histogram h;

        for (int64_t i = 0; i < 1001; ++i) {
            h.insert(i);
        }

        for (int64_t i = 0; i < 11; ++i) {
            tassert(h.bucket_counts()[i] == 1);
        }

        for (int64_t i = 11; i < 20; ++i) {
            tassert(h.bucket_counts()[i] == 10);
        }

        for (int64_t i = 20; i < 29; ++i) {
            tassert(h.bucket_counts()[i] == 100);
        }
    }

    {
        Histogram h;

        for (int64_t i = 0; i < 1000; ++i) {
            h.insert(1);
        }

        for (int64_t i = 0; i < 29; ++i) {
            if (i != 1) {
                tassert(h.bucket_counts()[i] == 0);
            }
            else {
                tassert(h.bucket_counts()[i] == 1000);
            }
        }
    }
}


static void
run(const Kernel::kernels_t & kernels)
{
    test_will_add_overflow();
    test_will_mult_overflow();
    test_mtx_reader();
    test_read_mtx_data();
    test_histogram();
    test_mtx_is_symmetric<double, int>();
#ifdef WITH_SAMPLE_HPCG
    test_mtx_gen_hpcg();
#endif
    test_kernels(kernels);

    tlog("\n%s  %ld/%ld test failed.\n",
         g_tests_failed == 0 ? "OK" : "ERROR",
         g_tests_failed,
         g_tests_total);
}

} // namespace test
