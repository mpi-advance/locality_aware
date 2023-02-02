#!/usr/bin/bash
#SBATCH --output ../../benchmark_tests/standard_torsten/D_10_Hopper_Standard_one_node
#SBATCH --error ../../benchmark_tests/standard_torsten/D_10_Hopper_Standard_one_node_err
#SBATCH --partition general
#SBATCH --ntasks=32
#SBATCH --nodes=1

#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=ageyko@unm.edu

module load openmpi

srun --partition=general --nodes=1 --ntasks=2 --time=24:00:00 ../../build/benchmarks/torsten_standard_comm ../../test_data/D_10.pm 20 D_10 STANDARD
srun --partition=general --nodes=1 --ntasks=4 --time=24:00:00 ../../build/benchmarks/torsten_standard_comm ../../test_data/D_10.pm 20 D_10 STANDARD
srun --partition=general --nodes=1 --ntasks=8 --time=24:00:00 ../../build/benchmarks/torsten_standard_comm ../../test_data/D_10.pm 20 D_10 STANDARD
srun --partition=general --nodes=1 --ntasks=16 --time=24:00:00 ../../build/benchmarks/torsten_standard_comm ../../test_data/D_10.pm 20 D_10 STANDARD
srun --partition=general --nodes=1 --ntasks=32 --time=24:00:00 ../../build/benchmarks/torsten_standard_comm ../../test_data/D_10.pm 20 D_10 STANDARD
