#include "spmv.h"


// SpMV kernels for computing  y = A * x, where A is a sparse matrix
// represented by different formats.

/**
 * Kernel for CSR format.
 */
template <typename VT, typename IT>
static void
spmv_omp_csr(const ST num_rows,
             const IT * RESTRICT row_ptrs,
             const IT * RESTRICT col_idxs,
             const VT * RESTRICT values,
             const VT * RESTRICT x,
             VT * RESTRICT y)
{
    #pragma omp parallel for schedule(static)
    for (ST row = 0; row < num_rows; ++row) {
        VT sum{};
        for (IT j = row_ptrs[row]; j < row_ptrs[row + 1]; ++j) {
            sum += values[j] * x[col_idxs[j]];
        }
        y[row] += sum;
    }
}


/**
 * Kernel for ELL format, data structures use row major (RM) layout.
 */
template <typename VT, typename IT>
static void
spmv_omp_ell_rm(
        const ST num_rows,
        const ST nelems_per_row,
        const IT * RESTRICT col_idxs,
        const VT * RESTRICT values,
        const VT * RESTRICT x,
        VT * RESTRICT y)
{
    #pragma omp parallel for schedule(static)
    for (ST row = 0; row < num_rows; row++) {
        VT sum{};
        for (ST i = 0; i < nelems_per_row; i++) {
            VT val = values[  row * nelems_per_row + i];
            IT col = col_idxs[row * nelems_per_row + i];

            sum += val * x[col];
        }
        y[row] += sum;
    }
}

/**
 * Kernel for ELL format, data structures use column major (CM) layout.
 */
template <typename VT, typename IT>
static void
spmv_omp_ell_cm(
        const ST num_rows,
        const ST nelems_per_row,
        const IT * RESTRICT col_idxs,
        const VT * RESTRICT values,
        const VT * RESTRICT x,
        VT * RESTRICT y)
{
    #pragma omp parallel for schedule(static)
    for (ST row = 0; row < num_rows; row++) {
        VT sum{};
        for (ST i = 0; i < nelems_per_row; i++) {
            VT val = values[row + i * num_rows];
            IT col = col_idxs[row + i * num_rows];

            sum += val * x[col];
        }
        y[row] += sum;
    }
}


/**
 * Kernel for Sell-C-Sigma. Supports all Cs > 0.
 */
template <typename VT, typename IT>
static void
spmv_omp_scs(const ST C,
             const ST n_chunks,
             const IT * RESTRICT chunk_ptrs,
             const IT * RESTRICT chunk_lengths,
             const IT * RESTRICT col_idxs,
             const VT * RESTRICT values,
             const VT * RESTRICT x,
             VT * RESTRICT y)
{

    #pragma omp parallel for schedule(static)
    for (ST c = 0; c < n_chunks; ++c) {
        VT tmp[C];
        for (ST i = 0; i < C; ++i) {
            tmp[i] = VT{};
        }

        IT cs = chunk_ptrs[c];

        // TODO: use IT wherever possible
        for (IT j = 0; j < chunk_lengths[c]; ++j) {
            for (IT i = 0; i < (IT)C; ++i) {
                tmp[i] += values[cs + j * (IT)C + i] * x[col_idxs[cs + j * (IT)C + i]];
            }
        }

        for (ST i = 0; i < C; ++i) {
            y[c * C + i] += tmp[i];
        }
    }
}


//          name      function         is_gpu format
REG_KERNELS("scs",    spmv_omp_scs,    false, MatrixFormat::SellCSigma);
REG_KERNELS("ell-rm", spmv_omp_ell_rm, false, MatrixFormat::EllRm);
REG_KERNELS("ell-cm", spmv_omp_ell_cm, false, MatrixFormat::EllCm);
REG_KERNELS("csr",    spmv_omp_csr,    false, MatrixFormat::Csr);
