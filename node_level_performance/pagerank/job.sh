#!/bin/bash -l
#SBATCH --job-name=spmv
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --constraint="gpu&perf"
#SBATCH --account=class07

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1

srun ./pagerank_csr.exe soc-Epinions1.mtx
gprof pagerank_csr.exe

