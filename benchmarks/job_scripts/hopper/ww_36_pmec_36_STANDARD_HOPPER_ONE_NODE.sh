#!/usr/bin/bash
#SBATCH --output ../../../benchmark_tests/standard_torsten/ww_36_pmec_36_Hopper_Standard_one_node
#SBATCH --error ../../../benchmark_tests/standard_torsten/ww_36_pmec_36_Hopper_Standard_one_node_err
#SBATCH --partition general
#SBATCH --ntasks=32
#SBATCH --nodes=1

#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=ageyko@unm.edu

module load openmpi

srun --partition=general --nodes=1 --ntasks=2 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 5 ww_36_pmec_36 STANDARD
srun --partition=general --nodes=1 --ntasks=4 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 5 ww_36_pmec_36 STANDARD
srun --partition=general --nodes=1 --ntasks=8 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 5 ww_36_pmec_36 STANDARD
srun --partition=general --nodes=1 --ntasks=16 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 5 ww_36_pmec_36 STANDARD
srun --partition=general --nodes=1 --ntasks=32 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 5 ww_36_pmec_36 STANDARD
