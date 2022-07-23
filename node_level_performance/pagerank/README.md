Experimenting with a pagerank implementation

## Building 

The `makefile` is very simple. Running `make` will generate

+ `pagerank_csr.exe` implementation with CSR (Compressed Sparse Row) matrix storage format
+ `pagerank_csc.exe` implementation with CSC (Compressed Sparse Column) storage format (for reference)

The compiler options include `-pg`, which instruments the code for profiling. Also, function inlining is disabled so you can actually see interesting profiling data.


## Usage on Piz Daint

+ `srun ./pagerank_csr.exe <matrixfile>`

After program termination, you will see that a file named `gmon.out` was generated in the current directory. It contains profiling information, i.e., how much time was spent in the different functions of the code. In order to read this info, you have to use the `gprof` tool:

+ `gprof pagerank_csr.exe`

This will produce a human-readable output with a "flat profile" (this is what we want to look at)  and a "butterfly profile" (which we can ignore here). The flat profile tells you which functions take most of the execution time.

## Exercise

1. Log into the Piz Daint frontend. Either start an interactive job (with, e.g., `salloc -N 1 --time=01:00:00`) or use the `job.sh` script provided and modify as needed. 
2. Execute the `pagerank_csr` code with the following matrices (all located in ):
   + `soc-Epinions1.mtx`
   + `foo.mtx`
   + `bar.mtx`
   What is, in general, the execution "hot spot," i.e., which function takes most of the runtime?
3. Parallelize the code with OpenMP diectives (hint: The relevant loops are all in the functions near the end of the source file). Load the `likwid` module and run it with
   + `env OMP_NUM_THREADS=12 OMP_PROC_BIND=close OMP_PLACES=cores srun ./pagerank_csr.exe <matrixfile>`
   What speedup do you get from 1 to 12 threads (`OMP_NUM_THREADS` setting)? Is this behavior expected?
4. The Haswell CPU in Piz Daint has a maximum memory bandwidth of about 62 Gbyte/s. What is the upper performance limit in Gflop/s for the sparse matrix-vector multiplication in the pagerank code?
5. Use the `likwid-perfctr` tool to find out whether this limit is actually attained. You can either instrument the code yourself with markers or use the instrumented version we provide in the `likwid/` folder.
