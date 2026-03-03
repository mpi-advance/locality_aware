#!/bin/bash
  
#SBATCH --output=reorder_n2.%j.out
#SBATCH --error=reorder_n2.%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=112
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --partition=pbatch

module load gcc
module load openmpi

cd ${HOME}/locality_fork/build/benchmarks
folder=/usr/workspace/bienz1/benchmark_mats
for mat in $folder/*.pm; do
    echo $mat
    for (( nodes = 2; nodes <= 2; nodes*=2 ));
    do
        procs=$((112*nodes))
        srun -n $procs -N $nodes ./reorder_msgs $mat
    done
done

