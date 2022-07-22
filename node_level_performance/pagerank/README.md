Experimenting with a pagerank implementation

## Building 

The `makefile` is very simple. Running `make` will generate

+ `pagerank_csr.exe` implementation with CSR (Compressed Sparse Row) matrix storage format
+ `pagerank_csc.exe` implementation with CSC (Compressed Sparse Column) storage format (for reference)

The compiler options include `-pg`, which instruments the code for profiling. Also, function inlining is disabled so you can actually see interesting profiling data.


## Usage

+ `./pagerank_csr.exe <matrixfile>`

After program termination, you will see that a file named `gmon.out` was generated in the current directory. It contains profiling information, i.e., how much time was spent in the different functions of the code. In order to read this info, you have to use the `gprof` tool:

+ `gprof pagerank_csr.exe`

This will produce a human-readable output with a "flat profile" (this is what we want to look at)  and a "butterfly profile" (which we can ignore here). The flat profile tells you which functions take most of the execution time.

## Exercise

1. Log into the Piz Daint frontend. Either start an interactive job or use the `job.sh` script provided.
2. Execute the `pagerank_csr` code with the following matrices:
   + `soc-Epinions1.mtx`
   + `foo.mtx`
   + `bar.mtx`
   What is, in general, 
