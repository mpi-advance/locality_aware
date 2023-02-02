#!/usr/bin/bash

#SBATCH --output ../../benchmark_tests/standard_torsten/D_10_Hopper_Standard_many_node
#SBATCH --error ../../benchmark_tests/standard_torsten/D_10_Hopper_Standard_many_node_err
#SBATCH --partition general
#SBATCH --ntasks=128
#SBATCH --nodes=4

#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=ageyko@unm.edu 

module load openmpi

srun --partition=general --nodes=1 --ntasks=64 --time=24:00:00 ../../build/benchmarks/torsten_standard_comm ../../test_data/D_10.pm 20 D_10 STANDARD 

srun --partition=general --nodes=1 --ntasks=128 --time=24:00:00 ../../build/benchmarks/torsten_standard_comm ../../test_data/D_10.pm 20 D_10 STANDARD 
