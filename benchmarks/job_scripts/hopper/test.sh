#!/usr/bin/bash
#SBATCH --output ../../../benchmark_tests/standard_torsten/ww_36_pmec_36/test_out
#SBATCH --error ../../../benchmark_tests/standard_torsten/ww_36_pmec_36/ww_36_pmec_36_Hopper_RMA_one_node_err
#SBATCH --open-mode=append#SBATCH 
#SBATCH --partition debug
#SBATCH --ntasks=32
#SBATCH --nodes=1

#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=ageyko@unm.edu

module load openmpi

srun --partition=debug --nodes=1 --ntasks=10 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
srun --partition=debug --nodes=1 --ntasks=11 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
srun --partition=debug --nodes=1 --ntasks=12 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
srun --partition=debug --nodes=1 --ntasks=13 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
srun --partition=debug --nodes=1 --ntasks=14 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
srun --partition=debug --nodes=1 --ntasks=15 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
