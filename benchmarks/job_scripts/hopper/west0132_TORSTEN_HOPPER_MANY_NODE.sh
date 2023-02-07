#!/usr/bin/bash
#SBATCH --output ../../../benchmark_tests/standard_torsten/west0132_Hopper_Torsten_many_node
#SBATCH --error ../../../benchmark_tests/standard_torsten/west0132_Hopper_Torsten_many_node_err
#SBATCH --partition general
#SBATCH --ntasks=64
#SBATCH --nodes=2

#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=ageyko@unm.edu

module load openmpi

srun --partition=general --nodes=2 --ntasks=48 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/west0132.pm 5 west0132 TORSTEN
srun --partition=general --nodes=2 --ntasks=64 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/west0132.pm 5 west0132 TORSTEN
