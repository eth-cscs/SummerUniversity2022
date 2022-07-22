Experimenting with SpMV code

This exercise lets youe experiment with a CUDA GPU implementation of sparse matrix-vector multiplication.

## Building

First, run `source ./modules.sh` in the SpMV folder. After that, running `make` will generate

+ `spmv-omp` OpenMP-parallelizied spmv on the CPU,
+ `spmv-gpu` CUDA implementaiton of spmv for the GPU

Build binaries separately:

+ `make spmv-omp`
+ `make spmv-gpu`

By default the CPU binary is built with gcc.

The GPU binary is always built by using `nvcc`.

## General Usage for GPUs

+ `./spmv-gpu <matrix-file> <matrix-format> <num-elems-per-row>` where `matrix-file` is the matrix market file to perform the SpMV with, `matrix-format` is the matrix format to convert to before performing the SpMV (choices include `csr` (for CSR) and `scs` (for SELL-C-sigma), defaulting to `csr`).
+ Options
  - `-c <C>` specify block size C for SELL-C-sigma
  - `-s <S>` specify sorting range sigma for SELL-C-sigma
  - `-dp` use double precision
  - `-sp` use single precision
  - `-no-verify` do not run result verfication
+ Example
  - `./spmv-gpu $MATRICES/DLR1.bmtx csr -dp -c 32 -s 32 -no-verify`
    runs SpMV in double precision with C=32 and sigma=32 on the DLR1 matrix.
    
## Testing


## Experiments

We want to know which matrices are "good" or "bad" on the GPU, i.e., which make good use of the resources and which do not. 


            dram_read_throughput:  Device memory read throughput
           dram_write_throughput:  Device memory write throughput
              l2_read_throughput:  Memory read throughput seen at L2 cache for all read requests
             l2_write_throughput:  Memory write throughput seen at L2 cache for all write requests
              achieved_occupancy:  Ratio of the average active warps per active cycle to the maximum number of warps supported on a multiprocessor
                 dram_read_bytes:  Total bytes read from DRAM to L2 cache
                dram_write_bytes:  Total bytes written from L2 cache to DRAM
