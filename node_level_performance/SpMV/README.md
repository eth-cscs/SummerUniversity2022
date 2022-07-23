Experimenting with SpMV code

This exercise lets you experiment with a CUDA GPU implementation of sparse matrix-vector multiplication on the GPUs of Piz Daint (NVIDIA P100).

## Building

First, run `source ./modules.sh` in the SpMV folder. After that, running `make` will generate

+ `spmv-omp` OpenMP-parallelizied spmv on the CPU,
+ `spmv-gpu` CUDA implementation of spmv for the GPU

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
  - `-no-verify` do not run result verification
    
## Testing
+ Example
  - `./spmv-gpu $MATRICES/HPCG-192-192-192.bmtx scs -dp -c 32 -s 32 -no-verify`
    runs SpMV in double precision with C=32 and sigma=32 on the DLR1 matrix.
  - The benchmark prints, among other things, data about the matrix (nnz, nnzr, variation of nnzr), the performance of the SpMV in Gflop/s, the beta factor, and the optimistic code balance for alpha=0 and alpha=1/Nnzr.
  - The run above is, as SpMV on the P100 GPU goes, as good as it gets. **Calculate the memory bandwidth drawn by the SpMV kernel from the available data.** This is now our empirical value for the memory bandwidth of the GPU.
  - `job.sh` is a simple job script that will help you get started. Submit it to the system with `sbatch job.sh`
  - **Hint**: The C value of 32 is rather optimal for GPUs, so you don't need to fiddle around with it. For sigma, we typically try powers of 2 from 32 to 4096.

## Experiments

We want to know which matrices are "good" or "bad" on the GPU, i.e., which make good use of the resources and which do not. Also we want to see which format is better - CSR or SELL-C-sigma.

+ Consider the following set of matrices. Note that you will not be able to run all of them with all combinations of parameters, so plan your experiments wisely:
  - HPCG-192-192-192
  - DLR1
  - Hamrle3
  - kkt_power
  - radom4M10
  - qcd5_4
  - com-Orkut
+ **Questions:**

1. Is CSR anywhere near competitive for any of the matrices?
  2. For which matrices is a larger sigma required to get better performance? Can you imagine why?
+ From performance measurements alone we cannot determine the actual memory bandwidth taken by the kernel or its actual code balance. We need a tool that gives us the memory bandwidth. The NVIDIA `nvprof` profiler can do this.
  - To use the profiler, instead of running the binary directly you run it with `nvprof` as a wrapper: `nvprof -m <metric> ./spmv-gpu <whatever>`
  - Many metrics are possible. This is a list of interesting metrics to consider:
    - `dram_read_bytes`:  Total bytes read from DRAM to L2 cache
    - `dram_write_bytes`:  Total bytes written from L2 cache to DRAM
    - `dram_read_throughput`:  Device memory read throughput
    - `dram_write_throughput`:  Device memory write throughput
    - `l2_read_throughput`:  Memory read throughput seen at L2 cache for all read requests
    - `l2_write_throughput`:  Memory write throughput seen at L2 cache for all write requests
    - `achieved_occupancy`:  Ratio of the average active warps per active cycle to the maximum number of warps supported on a multiprocessor
  - Example: `nvprof -m dram_read_throughput ./spmv-gpu $MATRICES/HPCG-192-192-192.bmtx scs -dp -c 32 -s 32 -no-verify`
  - The profiler gives you the metric _per kernel invocation_ as an average, minimum, and maximum value. **Important**: With the profiler active, the performance of the code is much reduced, so you won't get a proper performance reading from the code itself.
+ **Further questions:**

3. What is the actual (measured) code balance of SpMV when running the following  matrices? Does it agree with the output of the benchmark?
    - `random4M10`
    - `DLR1`
  4. How well do these two matrices utilize the memory bandwidth?

