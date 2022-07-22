Experimenting with SpMV code

## Building

Running `make` will generate

+ `spmv-omp` OpenMP parallelizied spmv on the CPU,
+ `spmv-gpu` CUDA implementaiton of spmv for the GPU

Build binaries separately:

+ `make spmv-omp`
+ `make spmv-gpu`

By default the CPU binary is built with gcc.

The GPU binary is always built by using `nvcc`.

## Usage

GPU:
+ `./spmv-gpu <matrix-file> <matrix-format> <num-elems-per-row>` where `matrix-file` is the matrix market file to perform the SpMV with, `matrix-format` is the matrix format to convert to before performing the SpMV (choices include `csr` and `ell`, defaulting to `csr`). `num-elems-per-row` is the number of elements per row in the matrix that is useful for ell matrix format.

CPU/OpenMP:
+ `./spmv-omp <matrix-file> <matrix-format> <num-elems-per-row>` where `matrix-file` is the matrix market file to perform the SpMV with, `matrix-format` is the matrix format to convert to before performing the SpMV (choices include `csr` and `ell`, defaulting to `csr`). `num-elems-per-row` is the number of elements per row in the matrix that is useful for ell matrix format.
+ might need to set environment variables accordingly:
  + `OMP_PROC_BIND=true|close|spread`
  + `OMP_PLACES=threads|cores|sockets|...`
  + `OMP_NUM_THREADS=<no. of threads>`
