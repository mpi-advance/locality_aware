#!/bin/bash
#SBATCH --job-name run_allreduce_ipc_copy_2nodes
#SBATCH --output slurm-%j-%x.out
#SBATCH --error slurm-%j-%x.err
#SBATCH -N 2
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --partition ghx4
#SBATCH --time 01:15:00
#SBATCH --account=bebi-dtai-gh

module load craype-accel-nvidia90
module unload gcc-native
module load gcc-native/12
export MPICH_GPU_SUPPORT_ENABLED=1

cd $HOME/locality_aware/build/src/heterogeneous/tests

n_nodes=2
gpus_per_node=4

echo "1 Processes Per GPU, ${gpus_per_node} GPUs Per Node"
srun -N ${n_nodes} --ntasks-per-node=4 --cpus-per-task=1 --gpus-per-node=${gpus_per_node} ./allreduce_gpu

for ppn in 8 16 32 64; do
    ppg=$(( ppn / gpus_per_node ))
    echo "${ppg} Processes Per GPU, ${gpus_per_node} GPUs Per Node"
    srun -N ${n_nodes} --ntasks-per-node=${ppn} --cpus-per-task=1 --gpus-per-node=${gpus_per_node} ./allreduce_mps_copy
done