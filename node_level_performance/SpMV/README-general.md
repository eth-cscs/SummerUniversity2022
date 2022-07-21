This example demonstrates the Sparse Matrix Vector product kernel on OpenMP and CUDA.  

## Building

Running `make` will generate

+ `spmv-omp` OpenMP parallelizied spmv on the CPU,
+ `spmv-gpu` CUDA implementaiton of spmv for the GPU

Build binaries separately:

+ `make spmv-omp`
+ `make spmv-gpu`

By default the CPU binary is build with gcc. Using the Intel compiler is
supported by specifying `COMPILER=intel` when running `make`:

```
make COMPILER=intel
```

The GPU binary is always build by using `nvcc`.

## Usage

GPU:
+ `./spmv-gpu <matrix-file> <matrix-format> <num-elems-per-row>` where `matrix-file` is the matrix market file to perform the SpMV with, `matrix-format` is the matrix format to convert to before performing the SpMV (choices include `csr` and `ell`, defaulting to `csr`). `num-elems-per-row` is the number of elements per row in the matrix that is useful for ell matrix format.

CPU/OpenMP:
+ `./spmv-omp <matrix-file> <matrix-format> <num-elems-per-row>` where `matrix-file` is the matrix market file to perform the SpMV with, `matrix-format` is the matrix format to convert to before performing the SpMV (choices include `csr` and `ell`, defaulting to `csr`). `num-elems-per-row` is the number of elements per row in the matrix that is useful for ell matrix format.
+ might need to set environment variables accordingly:
  + `OMP_PROC_BIND=true|close|spread`
  + `OMP_PLACES=threads|cores|sockets|...`
  + `OMP_NUM_THREADS=<no. of threads>`

*Making changes:*

It should be enough to change the kernels in `spmv-omp.cpp` and `spmv-gpu.cu`.

## TODO:

+ Reading `symmetric` matrices in matrix market format. Currently, only `general` matrices are allowed.
+ Support `std::complex<float|double>`? The original Homework 5 code dit that.
+ Currently there are separate binaries created for GPU and OpenMP, do we want
  to keep that?
+ There is propably room for code/structure improvments, feel free to
  add/change the code.
+ Any opinions on using `-Ofast`?
  + GCC seems to do not perform some optimizations with `-O3` that the Intel
    compiler does with `-O3`
