#!/bin/bash -l
#SBATCH --job-name=spmv
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --constraint="gpu&perf"
#SBATCH --account=class07

export CRAY_CUDA_MPS=1

source ./modules.sh

echo Working qcd5_4
srun ./spmv-gpu $MATRICES/qcd5_4.mtx csr -dp -c 32 -s 32 -no-verify
srun ./spmv-gpu $MATRICES/qcd5_4.mtx scs -dp -c 32 -s 32 -no-verify

