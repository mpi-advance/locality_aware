#!/bin/bash
#SBATCH --job-name=allreduce_N2
#SBATCH --output=allreduce_N2_%j.out
#SBATCH --error=allreduce_N2_%j.err
#SBATCH --ntasks-per-node=112
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --nodes=2

cd $HOME/locality_develop/build/benchmarks

srun --nodes=2 --ntasks-per-node=112 --cpus-per-task=1 ./allreduce

