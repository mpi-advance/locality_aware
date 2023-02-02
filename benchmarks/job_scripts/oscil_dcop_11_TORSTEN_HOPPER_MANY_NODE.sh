#!/usr/bin/bash
#SBATCH --output ../../benchmark_tests/standard_torsten/oscil_dcop_11_Hopper_Torsten_many_node
#SBATCH --error ../../benchmark_tests/standard_torsten/oscil_dcop_11_Hopper_Torsten_many_node_err
#SBATCH --partition general
#SBATCH --ntasks=128
#SBATCH --nodes=4

#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=ageyko@unm.edu

module load openmpi

srun --partition=general --nodes=2 --ntasks=64 --time=24:00:00 ../../build/benchmarks/torsten_standard_comm ../../test_data/oscil_dcop_11.pm 20 oscil_dcop_11 TORSTEN
srun --partition=general --nodes=3 --ntasks=96 --time=24:00:00 ../../build/benchmarks/torsten_standard_comm ../../test_data/oscil_dcop_11.pm 20 oscil_dcop_11 TORSTEN
srun --partition=general --nodes=4 --ntasks=128 --time=24:00:00 ../../build/benchmarks/torsten_standard_comm ../../test_data/oscil_dcop_11.pm 20 oscil_dcop_11 TORSTEN
