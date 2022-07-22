Experimenting with a pagerank implementation

## Building 

The `makefile` is very simple. Running `make` will generate

+ `pagerank_csr.exe` implementation with CSR matrix storage format
+ `pagerank_csc.exe` implementaion with CSC storage format (for reference)

The compiler options include `-pg`, which instruments the code for profiling. 


## Usage

+ `./pagerank_csr.exe <matrixfile>`

After program termination, you will see that a file named `gmon.out` was generated in the current directory. It contains profiling information, i.e., how much time was spent in the different functions of the code. In order to read this info, you have to use the `gprof` tool:

+ `gprof pagerank_csr.exe`

This will produce a human-readable output with a "flat profile" (this is what we want to look at)  and a "butterfly profile" (which we can ignore here). 
