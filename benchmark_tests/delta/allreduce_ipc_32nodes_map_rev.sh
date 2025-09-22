#!/bin/bash
#SBATCH --job-name run_allreduce_ipc_32nodes_map_rev
#SBATCH --output slurm-%j-%x.out
#SBATCH --error slurm-%j-%x.err
#SBATCH -N 32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --partition gpuA100x4
#SBATCH --time 00:28:00
#SBATCH --account=bebi-delta-gpu

module load cuda/12.4.0
module load openmpi/5.0.5+cuda

cd $HOME/locality_aware/build/src/heterogeneous/tests

export CUDA_VISIBLE_DEVICES=0,1,2,3

mpirun --map-by ppr:1:numa --bind-to core --rank-by slot --display-map --display-allocation --report-bindings ./allreduce_gpu r

mpirun --map-by ppr:2:numa --bind-to core --rank-by slot --display-map --display-allocation --report-bindings ./allreduce_mps r
mpirun --map-by ppr:4:numa --bind-to core --rank-by slot --display-map --display-allocation --report-bindings ./allreduce_mps r
mpirun --map-by ppr:8:numa --bind-to core --rank-by slot --display-map --display-allocation --report-bindings ./allreduce_mps r
mpirun --map-by ppr:16:numa --bind-to core --rank-by slot --display-map --display-allocation --report-bindings ./allreduce_mps r
