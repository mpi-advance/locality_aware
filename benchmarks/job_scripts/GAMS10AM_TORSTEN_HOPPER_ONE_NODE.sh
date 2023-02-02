#!/usr/bin/bash

#SBATCH --output ../../benchmark_tests/standard_torsten/GAMS10AM_Hopper_Torsten_one_node
#SBATCH --error ../../benchmark_tests/standard_torsten/GAMS10AM_Hopper_Torsten_one_node_err
#SBATCH --partition general
#SBATCH --ntasks=32
#SBATCH --nodes=1

#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=ageyko@unm.edu 

module load openmpi

for i in {1..3}
do
  let x=2**i
  srun --partition=general --nodes=1 --ntasks=$x --time=24:00:00 ../../build/benchmarks/torsten_standard_comm ../../test_data/gams10am.pm 20 gams10am TORSTEN 
done

srun --partition=general --nodes=1 --ntasks=16 --time=24:00:00 ../../build/benchmarks/torsten_standard_comm ../../test_data/gams10am.pm 20 gams10am TORSTEN 

srun --partition=general --nodes=1 --ntasks=32 --time=24:00:00 ../../build/benchmarks/torsten_standard_comm ../../test_data/gams10am.pm 20 gams10am TORSTEN 
