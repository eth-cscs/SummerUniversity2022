#!/bin/bash -l
#SBATCH --job-name=spmv
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --constraint="gpu&perf"
#SBATCH --account=class07

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1

source ./modules.sh

srun ./spmv-gpu /scratch/snx3000/class355/ghager/matrices/HPCG-192-192-192.bmtx scs -dp -c 32 -s 32 -no-verify
