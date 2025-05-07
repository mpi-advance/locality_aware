#!/bin/bash
#SBATCH --job-name run_allreduce_ipc_2nodes
#SBATCH --output slurm-%j-%x.out
#SBATCH --error slurm-%j-%x.err
#SBATCH -N 2
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --partition gpuA100x4
#SBATCH --time 01:15:00
#SBATCH --account=bebi-delta-gpu

module load cuda/12.4.0
module load openmpi/5.0.5+cuda

cd $HOME/locality_aware/build/src/heterogeneous/tests

export CUDA_VISIBLE_DEVICES=0,1,2,3

mpirun -N 4 ./allreduce_gpu

mpirun -N 8 ./allreduce_mps
mpirun -N 16 ./allreduce_mps
mpirun -N 32 ./allreduce_mps
mpirun -N 64 ./allreduce_mps
