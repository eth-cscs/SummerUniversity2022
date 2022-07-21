#include "spmv.h"

// SpMV kernels for computing  y = A * x, where A is a sparse matrix
// represented by different formats.

/**
 * Kernel for CSR format.
 */
template <typename VT, typename IT>
__global__ __launch_bounds__(default_block_size)
static void
spmv_csr(const ST num_rows,
         const IT * RESTRICT row_ptrs,
         const IT * RESTRICT col_idxs,
         const VT * RESTRICT values,
         const VT * RESTRICT x,
         VT * RESTRICT y)
{
    ST row = threadIdx.x + blockDim.x * blockIdx.x;

    if (row < num_rows) {
        VT sum{};
        for (IT j = row_ptrs[row]; j < row_ptrs[row + 1]; ++j) {
            sum += values[j] * x[col_idxs[j]];
        }
        y[row] += sum;
    }
}

/**
 * Kernel for ELL format, data structures use column major (CM) layout.
 */
template <typename VT, typename IT>
__global__ __launch_bounds__(default_block_size)
static void
spmv_ell_cm(const ST num_rows,
         const ST nelems_per_row,
         const IT * RESTRICT col_idxs,
         const VT * RESTRICT values,
         const VT * RESTRICT x,
         VT * RESTRICT y)
{
    ST row = threadIdx.x + blockDim.x * blockIdx.x;

    if (row < num_rows) {
        VT sum{};
        for (ST i = 0; i < nelems_per_row; ++i) {
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
__global__
static void
spmv_scs(const ST C,
         const ST n_chunks,
         const IT * RESTRICT chunk_ptrs,
         const IT * RESTRICT chunk_lengths,
         const IT * RESTRICT col_idxs,
         const VT * RESTRICT values,
         const VT * RESTRICT x,
         VT * RESTRICT y)
{
    ST row = threadIdx.x + blockDim.x * blockIdx.x;
    IT c   = row / C;  // the no. of the chunk
    IT idx = row % C;  // index inside the chunk

    if (row < n_chunks * C) {
        VT tmp{};
        IT cs = chunk_ptrs[c];

        for (IT j = 0; j < chunk_lengths[c]; ++j) {
            tmp += values[cs + j * C + idx] * x[col_idxs[cs + j * C + idx]];
        }

        y[row] += tmp;
    }

}


//          name      function     is_gpu format
REG_KERNELS("csr",    spmv_csr,    true,  MatrixFormat::Csr);
REG_KERNELS("ell-cm", spmv_ell_cm, true,  MatrixFormat::EllCm);
REG_KERNELS("scs",    spmv_scs,    true,  MatrixFormat::SellCSigma);
