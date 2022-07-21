#pragma once

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

template <typename VT, typename IT>
struct MtxData
{
    ST n_rows{};
    ST n_cols{};
    ST nnz{};

    bool is_sorted{};
    bool is_symmetric{};

    std::vector<IT> I;
    std::vector<IT> J;
    std::vector<VT> values;
};


class MtxReader
{
public:
    MtxReader() = default;

    explicit MtxReader(std::string file_name)
        : file_name_(file_name)
    {
        read_header();
    }

    explicit MtxReader(std::istream & input)
    {
        read_header(input);
    }


    MtxReader(const MtxReader &) = delete;
    MtxReader& operator=(const MtxReader &) = delete;

    MtxReader(MtxReader &&) = default;
    MtxReader& operator=(MtxReader &&) = default;

    ~MtxReader() = default;


    uint64_t m() const   { return m_; }
    uint64_t n() const   { return n_; }
    uint64_t nnz() const { return nnz_; }

    bool is_data_type_present() const { return is_data_type_present_; }
    bool is_real()              const { return is_real_; }
    bool is_complex()           const { return is_complex_; }
    bool is_integer()           const { return is_integer_; }

    bool is_general()           const { return is_general_; }
    bool is_symmetric()         const { return is_symmetric_; }

    std::string file_name()     const { return file_name_; }

    std::string comments()      const { return comments_; }
    std::string header_line()   const { return header_line_; }

    void new_data_stream(std::ifstream & f ) const {
        f.open(file_name());
        f.seekg(pos_);
    }

private:

    void read_header()
    {
        std::ifstream f(file_name_);
        if (!f.is_open()) {
            throw std::invalid_argument("Could not open file.");
        }
        read_header(f);
    }

    void read_header(std::istream & input)
    {
        // Process header line.
        {
            if (!std::getline(input, header_line_)) {
                throw std::invalid_argument("getline failed.");
            }

            std::istringstream s(header_line_);

            std::string banner;
            s >> banner;

            if (banner != "%%MatrixMarket") {
                throw std::invalid_argument("File is missing matrix market banner.");
            }

            std::string object;
            s >> object;

            if (object != "matrix") {
                throw std::invalid_argument("Only matrices are supported.");
            }

            std::string format;
            s >> format;

            if (format != "coordinate") {
                throw std::invalid_argument("Only coordinate format is supported.");
            }

            std::string data_type;
            s >> data_type;

            if (data_type != "") {
                is_data_type_present_ = true;
                is_real_    = (data_type == "real");
                is_complex_ = (data_type == "complex");
                is_integer_ = (data_type == "integer");

                if (!is_real_ && !is_complex_ && !is_integer_) {
                    throw std::invalid_argument("Unsupported data type detected.");
                }
            }

            std::string shape;
            s >> shape;

            is_symmetric_ = (shape == "symmetric");
            is_general_   = (shape == "general" || shape == "");

            if (!is_general_ && !is_symmetric_) {
                throw std::invalid_argument("Only general or symmetric matrices are supported.");
            }

        }

        // TODO: After the first line blank lines are allowed
        //       everywhere.

        // Process zero or more lines with comments.
        {
            std::stringstream s;

            pos_ = input.tellg();
            bool all_comments_read = false;

            do {
                std::string line;
                if (!std::getline(input, line)) {
                    throw std::invalid_argument("Processing comments failed.");
                }

                if (line.size() == 0
                    || (line.size() > 0 && line[0] == '%')) {
                    pos_ = input.tellg();
                    s << line << '\n';
                }
                else {
                    input.seekg(pos_);
                    all_comments_read = true;
                }
            }
            while (!all_comments_read);

            comments_ = s.str();
        }

        // Read matrix dims and nnz.
        {
            std::string line;

            if (!std::getline(input, line)) {
                throw std::invalid_argument("Could not read matrix dimensions.");
            }

            std::istringstream s(line);
            s >> m_ >> n_ >> nnz_;
            pos_ = input.tellg();
        }
    }

    uint64_t nnz_{};
    uint64_t m_{};
    uint64_t n_{};

    bool is_data_type_present_ { false };
    bool is_real_              { false };
    bool is_complex_           { false };
    bool is_integer_           { false };

    bool is_symmetric_         { false };
    bool is_general_           { false };

    std::ifstream::pos_type pos_{-1};
    std::string file_name_;
    std::string comments_;
    std::string header_line_;
};


template <typename VT, typename IT>
static void
sort_mtx(MtxData<VT, IT> & mtx)
{
    std::vector<ST> perm(mtx.nnz);
    std::iota(begin(perm), end(perm), ST{0});

    auto & I = mtx.I;
    auto & J = mtx.J;
    auto & v = mtx.values;

    std::sort(begin(perm), end(perm),
              [&](ST idx1, ST idx2) {
                return (I[idx1] == I[idx2]) ?
                    J[idx1] < J[idx2] : I[idx1] < I[idx2];
              });

    for (ST i = 0; i < mtx.nnz; ++i) {

        ST cycle_start = perm[i];

        if (cycle_start != i) {

            IT tmpI = I[cycle_start], tmpJ = J[cycle_start];
            VT tmpV = v[cycle_start];

            for (ST cur = cycle_start, next = perm[cur];
                 next != cycle_start;
                 cur = next, next = perm[next]) {

                I[cur] = I[next];
                J[cur] = J[next];
                v[cur] = v[next];

                perm[cur] = cur;
            }

            I[i] = tmpI; J[i] = tmpJ; v[i] = tmpV;
        }
   }
}

template <typename VT, typename IT>
static void
print_mtx(MtxData<VT, IT> & mtx)
{
    for (ST i = 0; i < mtx.nnz; ++i) {
        printf("[%ld]  %ld  %ld  %e\n",
               (long)i, (long)mtx.I[i], (long)mtx.J[i],
               (double)mtx.values[i]);
    }
}


template <typename VT, typename IT>
static bool
is_mtx_sorted(const MtxData<VT, IT> & mtx)
{
    // check if sorted
    bool is_sorted = true;

    for (ST i = 1; i < mtx.nnz; ++i) {
        if (mtx.I[i] == mtx.I[i - 1]) {
            if (mtx.J[i] == mtx.J[i - 1]) {
                fprintf(stderr, "ERROR: duplicated nnz in mtx file found.\n");
                exit(EXIT_FAILURE);
            }
            else if (mtx.J[i] < mtx.J[i - 1]) {
                is_sorted = false;
                break;
            }
        }
        else if (mtx.I[i] < mtx.I[i - 1]) {
            is_sorted = false;
            break;
        }
    }

    return is_sorted;
}

template <typename VT, typename IT>
static bool
is_mtx_symmetric(const MtxData<VT, IT> & mtx)
{
    bool is_symmetric = true;
    std::vector<int> checked(mtx.nnz);

    for (ST i = 0; i < mtx.nnz; ++i) {
        if (checked[i]) {
            continue;
        }

        const IT row = mtx.I[i];
        const IT col = mtx.J[i];

        bool found = false;

        // start with j = i, so that we also "find" the nnz on the diagonal
        for (ST j = i; j < mtx.nnz; ++j) {
            if (mtx.I[j] == col && mtx.J[j] == row) {
                checked[j] = 1;
                found = true;
                break;
            }
        }

        if (!found) {
            is_symmetric = false;
            break;
        }
    }

    return is_symmetric;
}

/**
 * @brief Check if @p mtx is symmetric, temporarily doubles the memory
 * allocated by mtx.
 *
 * Use CSR data structures to lookup non-zeros.
 * Required additional memory: nnz * IT + 2 * (n_rows + 1) * IT.
 */
template <typename VT, typename IT>
static bool
is_mtx_symmetric_fast(const MtxData<VT, IT> & mtx)
{
    std::vector<IT> row_ptrs(mtx.n_rows + 1);
    std::vector<IT> col_idxs(mtx.nnz);

    // count nnz of each row
    for (ST i = 0; i < mtx.nnz; ++i) {
        // shift index by 1, to make computation of final row pointers
        // easier later.
        ++row_ptrs[mtx.I[i] + 1];
    }

    // compute final row pointers
    std::partial_sum(row_ptrs.begin(), row_ptrs.end(), row_ptrs.begin());

    {
        std::vector<IT> row_moving_ptrs = row_ptrs;

        for (ST i = 0; i < mtx.nnz; ++i) {
            const IT row = mtx.I[i];
            const IT col = mtx.J[i];

            col_idxs[row_moving_ptrs[row]] = col;
            ++row_moving_ptrs[row];

            if (row_moving_ptrs[row] > row_ptrs[row + 1]) {
                fprintf(stderr, "ERROR: is_mtx_symmetric_fast found an error in matrix.\n");
                exit(EXIT_FAILURE);
            }
        }
    }

    bool is_symmetric = true;

    for (ST i = 0; i < mtx.nnz; ++i) {
        const IT row = mtx.I[i];
        const IT col = mtx.J[i];

        const IT * begin = &col_idxs[row_ptrs[col]];
        const IT * end   = &col_idxs[row_ptrs[col + 1]];

        if (std::find(begin, end, row) == end) {
            is_symmetric = false;
            break;
        }
    }

    return is_symmetric;
}

#ifndef WITH_SAMPLE_HPCG
#define WITH_SAMPLE_HPCG
#endif

#ifdef WITH_SAMPLE_HPCG

#define SAMPLE_PREFIX_HPCG "gen:hpcg:"

namespace mtx_gen_hpcg
{

static bool
starts_with(const char * whole_string, const char * prefix)
{
    if (whole_string == NULL || prefix == NULL) {
        return false;
    }

    std::string whole_string_str{ whole_string };
    std::string prefix_str{ prefix };

    if (whole_string_str.size() < prefix_str.size()) {
        return false;
    }

    return std::equal(prefix_str.begin(), prefix_str.end(), whole_string_str.begin());
}

static bool
is_sample_hpcg(const char * file_name)
{
    return starts_with(file_name, SAMPLE_PREFIX_HPCG);
}

static bool
get_dimensions(const char * description, uint64_t & nx, uint64_t & ny, uint64_t & nz)
{
    if (description == NULL) {
        fprintf(stderr, "ERROR: cannot create HPCG matrix from NULL description\n");
        exit(EXIT_FAILURE);
    }

    std::string desc{ description };

    if (starts_with(description, SAMPLE_PREFIX_HPCG)) {
        desc = desc.substr(std::string{ SAMPLE_PREFIX_HPCG }.size());
    }

    std::istringstream stream{ desc };
    std::string token;

    int64_t tmp;

    if (!std::getline(stream, token, ':')) return false;
    std::istringstream{token} >> tmp;
    if (tmp <= 0) return false;
    nx = tmp;

    if (!std::getline(stream, token, ':')) {
        nz = ny = nx;
        return true;
    }
    std::istringstream{token} >> tmp;
    if (tmp <= 0) return false;
    ny = tmp;

    if (!std::getline(stream, token, ':')) {
        nz = ny;
        return true;
    }

    std::istringstream{token} >> tmp;
    if (tmp <= 0) return false;
    nz = tmp;

    return true;
}

static ST
get_nnz(ST nx, ST ny, ST nz)
{
    ST nnz{};

    // inner volume
    nnz  = (nx - 2) * (ny - 2) * (nz - 2) * 27;
    // top and bottom
    nnz += (nx - 2) * (ny - 2) * 18 * 2;
    // front and back
    nnz += (nx - 2) * (nz - 2) * 18 * 2;
    // left and right
    nnz += (ny - 2) * (nz - 2) * 18 * 2;
    // edges along the x axis (w/o corners)
    nnz += (nx - 2) * 12 * 4;
    // edges along the y axis (w/o corners)
    nnz += (ny - 2) * 12 * 4;
    // edges along the z axis (w/o corners)
    nnz += (nz - 2) * 12 * 4;
    // 8 corners
    nnz += 8 * 8;

    return nnz;
}

static void
get_mtx_dimensions(const char * file_name, uint64_t & n_rows, uint64_t & n_cols, uint64_t & nnz)
{
    uint64_t nx, ny, nz;

    get_dimensions(file_name, nx, ny, nz);

    nnz = get_nnz(nx, ny, nz);

    n_rows = n_cols = nx * ny * nz;
}



static std::vector<std::tuple<ST, ST, ST>>
stencil_offsets()
{
    std::vector<std::tuple<ST, ST, ST>> offsets;
    offsets.reserve(9);

    for (int sz = -1; sz < 2; ++sz) {
        for (int sy = -1; sy < 2; ++sy) {
            for (int sx = -1; sx < 2; ++sx) {
                offsets.emplace_back(std::forward_as_tuple(sx, sy, sz));
            }
        }
    }

    return offsets;
}

/**
 * @brief Generate an HPCG matrix from @p description.
 *
 * @param description  Must be in the format of [gen:hpcg:]<nx>[:<ny>[:<nz>]]"
 *                     If ny an nz are omitted, ny = nz = nx is used, if nz is
 *                     omitted nz = ny is used.
 */
template <typename VT, typename IT>
static MtxData<VT, IT>
generate_mtx(const char * description)
{
    if (description == NULL) {
        fprintf(stderr, "ERROR: cannot create HPCG matrix from NULL description\n");
        exit(EXIT_FAILURE);
    }

    uint64_t tx, ty, tz;

    if (!get_dimensions(description, tx, ty, tz)) {
        fprintf(stderr, "ERROR: sample arguments are not of the form x:y:z\n");
        exit(EXIT_FAILURE);
    }

    ST nx = tx;
    ST ny = ty;
    ST nz = tz;

    ST n_rows = nx * ny * nz;
    ST nnz    = get_nnz(nx, ny, nz);

    MtxData<VT, IT> mtx;

    mtx.n_cols = n_rows;
    mtx.n_rows = n_rows;
    mtx.nnz    = nnz;

    mtx.is_sorted    = true;
    mtx.is_symmetric = true;

    mtx.I.reserve(nnz);
    mtx.J.reserve(nnz);
    mtx.values.reserve(nnz);


    std::vector<std::tuple<ST, ST, ST>> offsets = stencil_offsets();

    // FILE * fd = stderr;
    // fprintf(fd, "#hpcg: %%MatrixMarket matrix coordinate real general\n");
    // fprintf(fd, "#hpcg: %lu %lu %lu\n", (unsigned long)n_rows, (unsigned long)n_rows, (unsigned long)nnz);

    for (ST z = 0; z < nz; ++z) {
        ST els_z = z * nx * ny;
        for (ST y = 0; y < ny; ++y) {
            ST els_y = y * nx;
            for (ST x = 0; x < nx; ++x) {
                ST row = els_z + els_y + x;

                for (const auto & xyz : offsets) {
                    // If ST is unsigned, we wrap around, but as long as n{x,y,z}
                    // does not get too large we are still save.
                    ST neigh_x = x + std::get<0>(xyz);
                    ST neigh_y = y + std::get<1>(xyz);
                    ST neigh_z = z + std::get<2>(xyz);

                    if (neigh_x < 0 || neigh_x >= nx ||
                        neigh_y < 0 || neigh_y >= ny ||
                        neigh_z < 0 || neigh_z >= nz) {
                        continue;
                    }

                    ST col = neigh_z * ny * nx + neigh_y * nx + neigh_x;

                    VT value = (col != row ? VT{-1.0} : VT{26.0});

                    // fprintf(fd, "#hpcg: %lu %lu %d\n", (unsigned long)row, (unsigned long)col, (int)value);

                    mtx.I.emplace_back(row);
                    mtx.J.emplace_back(col);
                    mtx.values.emplace_back(value);

                }
            }
        }
    }

    return mtx;
}

} // namespace sample_hpcg
#endif // WITH_SAMPLE_HPCG

namespace bmtx
{
    struct header_t
    {
        uint64_t endian_marker;
        uint64_t version;
        uint64_t header_size;
        int64_t n_rows;
        int64_t n_cols;
        int64_t nnz;
        int64_t index_type_size;
        int64_t value_type_size;
        int64_t flags;
    };

    enum class flags : int64_t
    {
        is_symmetric = 0x01
    };

    template <typename T>
    struct is_fwi : std::false_type {};

    template <> struct is_fwi<uint64_t> : std::true_type {};
    template <> struct is_fwi<uint32_t> : std::true_type {};
    template <> struct is_fwi<uint16_t> : std::true_type {};

    template <> struct is_fwi<int64_t>  : std::true_type {};
    template <> struct is_fwi<int32_t>  : std::true_type {};
    template <> struct is_fwi<int16_t>  : std::true_type {};

    template <typename T, bool = std::is_arithmetic<T>::value || is_fwi<T>::value>
    struct is_arithmetic_or_fwi : std::false_type {};

    template <typename T>
    struct is_arithmetic_or_fwi<T, true> : std::true_type {};

    template <typename T,
              typename std::enable_if<is_arithmetic_or_fwi<T>::value, bool>::type = true>
    static void
    byte_swap(T & data)
    {
        constexpr size_t size_T{ sizeof(T) };
        char buffer_in[size_T];
        memcpy(buffer_in, &data, size_T);

        char buffer_out[size_T];
        for (size_t i = 0; i < size_T; ++i) {
            buffer_out[i] = buffer_in[size_T - 1 - i];
        }

        memcpy(&data, buffer_out, size_T);

        return;
    }

    void
    byte_swap(header_t & header)
    {
        #define SWAP_MEMBER(struct_data, member) \
            byte_swap(struct_data.member)

        #define HEADER_MEMBERS \
            X(endian_marker) X(version) X(header_size) \
            X(n_rows) X(n_cols) X(nnz) \
            X(index_type_size) X(value_type_size) \
            X(flags)

        #define X(member) SWAP_MEMBER(header, member);
        HEADER_MEMBERS
        #undef X

        #undef SWAP_MEMBER
        #undef HEADER_MEMBERS
    }

    template <typename T>
    void
    byte_swap(std::vector<T> & vector)
    {
        for (auto & v : vector) {
            byte_swap(v);
        }
    }
}

static bool
is_bmtx_file(const char * file_name)
{
    if (file_name == nullptr) {
        return false;
    }

    std::string file(file_name);
    std::string bmtx { "bmtx" };

    if (file.size() < bmtx.size()) {
        return false;
    }

    return std::equal(bmtx.rbegin(), bmtx.rend(), file.rbegin());
}

template <typename VT, typename IT>
static MtxData<VT, IT>
read_mtx_data_binary(const char * file_name)
{
    bmtx::header_t h;

    FILE * f = fopen(file_name, "rb");

    if (f == nullptr) {
        fprintf(stderr, "ERROR: opening %s failed.\n", file_name);
        exit(EXIT_FAILURE);
    }

    // Skip comments (all lines starting with '%') until we reach the header.
    {
        int c = fgetc(f);

        while (c == '%') {
            while ((c = fgetc(f)) != '\n');
            c = fgetc(f);
        }

        if (ungetc(c, f) != c) {
            fprintf(stderr, "ERROR: ungetc failed.\n");
            exit(EXIT_FAILURE);
        }
    }

    size_t ret = fread(&h, 1, sizeof(h), f);

    if (ret < sizeof(h)) {
        fprintf(stderr, "ERROR: reading header from %s failed.\n", file_name);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    bool perform_endian_conversion = h.endian_marker != uint64_t{1};

    if (perform_endian_conversion) {
        fprintf(stderr,
                "NOTE: bmtx with different endian detected, "
                "performing conversion... (should be 1, but is 0x%lx\n).\n",
                h.endian_marker);

        bmtx::byte_swap(h);

        if (h.endian_marker != uint64_t{1}) {
            fprintf(stderr,
                    "ERROR: endian conversion failed, should be 1, but is 0x%lx\n.\n",
                    h.endian_marker);
            fclose(f);
            exit(EXIT_FAILURE);
        }
    }

    if (h.header_size != sizeof(h)) {
        fprintf(stderr, "ERROR: bmtx header has unexpected size.\n");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    if (h.version != 1) {
        fprintf(stderr, "ERROR: bmtx header has unexpected version.\n");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    bool is_symmetric = h.flags & (int64_t)bmtx::flags::is_symmetric;

    MtxData<VT, IT> m;
    m.nnz = h.nnz;
    m.n_rows = h.n_rows;
    m.n_cols = h.n_cols;

    if (is_symmetric) {
        m.I.reserve(m.nnz * 2);
        m.J.reserve(m.nnz * 2);
        m.values.reserve(m.nnz * 2);
    }

    m.I.resize(m.nnz);
    m.J.resize(m.nnz);
    m.values.resize(m.nnz);

    if (   h.index_type_size != sizeof(int32_t)
        || h.value_type_size != sizeof(double)) {
        fprintf(stderr, "ERROR: bmtx file does not have index type = int32_t or value type = double\n");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    {
        using BMTX_IT = int32_t;

        std::vector<BMTX_IT> tmp(m.nnz);
        size_t to_read = m.nnz * sizeof(BMTX_IT);

        ret = fread(tmp.data(), 1, to_read, f);

        if (ret != to_read) {
            fprintf(stderr, "ERROR: reading less bytes than expected.\n");
            fclose(f);
            exit(EXIT_FAILURE);
        }

        for (size_t i = 0; i < tmp.size(); ++i) {
            m.I[i] = tmp[i];
        }

        if (perform_endian_conversion) {
            bmtx::byte_swap(m.I);
        }

        ret = fread(tmp.data(), 1, to_read, f);

        if (ret != to_read) {
            fprintf(stderr, "ERROR: reading less bytes than expected.\n");
            fclose(f);
            exit(EXIT_FAILURE);
        }

        for (size_t i = 0; i < tmp.size(); ++i) {
            m.J[i] = tmp[i];
        }

        if (perform_endian_conversion) {
            bmtx::byte_swap(m.J);
        }
    }

    {
        using BMTX_VT = double;
        std::vector<BMTX_VT> tmp(m.nnz);
        size_t to_read = m.nnz * sizeof(BMTX_VT);

        ret = fread(tmp.data(), 1, to_read, f);

        if (ret != to_read) {
            fprintf(stderr, "ERROR: reading less bytes than expected.\n");
            fclose(f);
            exit(EXIT_FAILURE);
        }

        for (size_t i = 0; i < tmp.size(); ++i) {
            m.values[i] = tmp[i];
        }

        if (perform_endian_conversion) {
            bmtx::byte_swap(m.values);
        }
    }

    fclose(f);

    if (is_symmetric) {
        for (ST i = 0; i < m.nnz; ++i) {
            if (m.I[i] != m.J[i]) {
                m.I.emplace_back(m.J[i]);
                m.J.emplace_back(m.I[i]);
                m.values.emplace_back(m.values[i]);
            }
        }

        m.nnz = m.values.size();
        m.is_symmetric = true;
    }

    m.is_sorted = is_mtx_sorted(m);

    return m;
}

template <typename VT, typename IT>
static MtxData<VT, IT>
read_mtx_data(std::istream & f,
              ST n_rows, ST n_cols,
              ST nnz, bool symmetric = false)
{
    MtxData<VT, IT> mtx;

    mtx.n_rows = n_rows;
    mtx.n_cols = n_cols;
    mtx.nnz    = nnz;

    if (symmetric) {
        mtx.I.reserve(mtx.nnz * 2);
        mtx.J.reserve(mtx.nnz * 2);
        mtx.values.reserve(mtx.nnz * 2);
    }

    mtx.I.resize(nnz);
    mtx.J.resize(nnz);
    mtx.values.resize(nnz);

    bool is_sorted = true;

    ST i = 0;

    while (i < nnz) {
        std::string line;

        if (!std::getline(f, line)) {
            std::fprintf(stderr, "ERROR: file seems to end early.\n");
            std::exit(EXIT_FAILURE);
        }

        if (line.empty()) {
            continue;
        }

        std::stringstream stream(line);

        stream >> mtx.I[i] >> mtx.J[i] >> mtx.values[i];

        if (mtx.I[i] < 1 || mtx.I[i] > n_rows) {
            fprintf(stderr, "ERROR: row index in mtx file is invalid.\n");
            exit(EXIT_FAILURE);
        }

        if (mtx.J[i] < 1 || mtx.J[i] > n_cols) {
            fprintf(stderr, "ERROR: column index in mtx file is invalid.\n");
            exit(EXIT_FAILURE);
        }

        mtx.I[i]--;  // adjust from 1-based to 0-based
        mtx.J[i]--;

        if (i > 0 && is_sorted) {
            if (mtx.I[i] == mtx.I[i - 1]) {
                if (mtx.J[i] == mtx.J[i - 1]) {
                    fprintf(stderr, "ERROR: duplicated nnz in mtx file found.\n");
                    exit(EXIT_FAILURE);
                }
                else if (mtx.J[i] < mtx.J[i - 1]) {
                    is_sorted = false;
                }
            }
            else if (mtx.I[i] < mtx.I[i - 1]) {
                is_sorted = false;
            }
        }

        ++i;
    }

    if ((size_t)nnz != mtx.values.size()) {
        fprintf(stderr, "ERROR: expected to read %lu values, but got %lu.\n",
                (size_t)nnz, mtx.values.size());
        exit(EXIT_FAILURE);
    }

    if (symmetric) {
        for (ST i = 0; i < nnz; ++i) {
            if (mtx.I[i] != mtx.J[i]) {
                mtx.I.emplace_back(mtx.J[i]);
                mtx.J.emplace_back(mtx.I[i]);
                mtx.values.emplace_back(mtx.values[i]);
            }
        }

        mtx.nnz = (ST)mtx.values.size();
        mtx.is_symmetric = true;

        is_sorted = false;
    }

    mtx.is_sorted = is_sorted;

    return mtx;
}

template <typename VT, typename IT>
static MtxData<VT, IT>
read_mtx_data_ascii(const char * file_name)
{
    MtxReader m(file_name);

    std::ifstream f;
    m.new_data_stream(f);

    return read_mtx_data<VT, IT>(f, m.m(), m.n(), m.nnz(), m.is_symmetric());
}

template <typename VT, typename IT>
static MtxData<VT, IT>
read_mtx_data(const char * file_name, bool sort)
{
    if (file_name == nullptr) {
        fprintf(stderr, "ERROR: file_name is NULL\n");
        exit(EXIT_FAILURE);
    }

    MtxData<VT, IT> mtx;

    log("reading mtx begin\n");

#ifdef WITH_SAMPLE_HPCG
    if (mtx_gen_hpcg::is_sample_hpcg(file_name)) {
        mtx = mtx_gen_hpcg::generate_mtx<VT, IT>(file_name);
    }
    else
#endif
    if (is_bmtx_file(file_name)) {
        mtx = read_mtx_data_binary<VT, IT>(file_name);
    }
    else {
        try {
            mtx = read_mtx_data_ascii<VT, IT>(file_name);
        }
        catch (std::exception & e) {
            fprintf(stderr, "ERROR: reading %s failed: %s\n",
                    file_name, e.what());
            exit(EXIT_FAILURE);
        }
    }

    if (sort && !mtx.is_sorted) {
        log("sorting mtx begin\n");
        sort_mtx(mtx);
        log("sorting mtx end\n");
        mtx.is_sorted = true;
    }

    if (!mtx.is_symmetric) {
        log("check symmetry mtx begin\n");
        mtx.is_symmetric = is_mtx_symmetric_fast(mtx);
        log("check symmetry mtx end\n");
    }

    log("reading mtx end\n");

    return mtx;
}


static void
get_mtx_dimensions_ascii(
        const char * file_name,
        uint64_t & n_rows, uint64_t & n_cols, uint64_t & nnz)
{
    MtxReader m(file_name);

    n_rows = m.m();
    n_cols = m.n();
    nnz    = m.nnz();

    return;
}

static void
get_mtx_dimensions_binary(
        const char * file_name,
        uint64_t & n_rows, uint64_t & n_cols, uint64_t & nnz)
{
    bmtx::header_t h;

    FILE * f = fopen(file_name, "rb");

    if (f == nullptr) {
        fprintf(stderr, "ERROR: opening %s failed.\n", file_name);
        exit(EXIT_FAILURE);
    }

    // Skip comments (all lines starting with '%') until we reach the header.
    {
        int c = fgetc(f);

        while (c == '%') {
            while ((c = fgetc(f)) != '\n');
            c = fgetc(f);
        }

        if (ungetc(c, f) != c) {
            fprintf(stderr, "ERROR: ungetc failed.\n");
            exit(EXIT_FAILURE);
        }
    }


    size_t ret = fread(&h, 1, sizeof(h), f);

    fclose(f);

    if (ret < sizeof(h)) {
        fprintf(stderr, "ERROR: reading header from %s failed.\n", file_name);
        exit(EXIT_FAILURE);
    }

    if (h.endian_marker != 1) {
        fprintf(stderr, "ERROR: bmtx with different endian detected.\n");
        exit(EXIT_FAILURE);
    }

    n_rows = h.n_rows;
    n_cols = h.n_cols;
    nnz    = h.nnz;

    return;
}


static void
get_mtx_dimensions(
        const char * file_name,
        uint64_t & n_rows, uint64_t & n_cols, uint64_t & nnz)
{
    if (file_name == nullptr) {
        fprintf(stderr, "ERROR: file_name is NULL\n");
        exit(EXIT_FAILURE);
    }

#ifdef WITH_SAMPLE_HPCG
    if (mtx_gen_hpcg::is_sample_hpcg(file_name)) {
        mtx_gen_hpcg::get_mtx_dimensions(file_name, n_rows, n_cols, nnz);
    }
    else
#endif
    if (is_bmtx_file(file_name)) {
        get_mtx_dimensions_binary(file_name, n_rows, n_cols, nnz);
    }
    else {
        get_mtx_dimensions_ascii(file_name, n_rows, n_cols, nnz);
    }
}


template <typename IT>
static bool
is_type_large_enough_for_mtx_sizes(const char * file_name)
{
    if (file_name == nullptr) {
        fprintf(stderr, "ERROR: file_name is NULL\n");
        exit(EXIT_FAILURE);
    }

    uint64_t n_rows{};
    uint64_t n_cols{};
    uint64_t nnz{};

#ifdef WITH_SAMPLE_HPCG
    if (mtx_gen_hpcg::is_sample_hpcg(file_name)) {
        mtx_gen_hpcg::get_mtx_dimensions(file_name, n_rows, n_cols, nnz);
    }
    else
#endif
    if (is_bmtx_file(file_name)) {
        get_mtx_dimensions_binary(file_name, n_rows, n_cols, nnz);
    }
    else {
        get_mtx_dimensions_ascii(file_name, n_rows, n_cols, nnz);
    }

    uint64_t it_max = std::numeric_limits<IT>::max();

    return n_rows <= it_max && n_cols <= it_max && nnz <= it_max;
}
