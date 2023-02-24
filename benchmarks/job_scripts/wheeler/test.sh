#!/usr/bin/bash
#SBATCH --output ../../../benchmark_tests/standard_torsten/ww_36_pmec_36/ww_36_pmec_36_Wheeler_RMA_one_node
#SBATCH --error ../../../benchmark_tests/standard_torsten/ww_36_pmec_36/ww_36_pmec_36_Wheeler_RMA_one_node_err
#SBATCH --open-mode=append
#SBATCH --partition debug
#SBATCH --ntasks=32
#SBATCH --nodes=4
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=ageyko@unm.edu

module load openmpi

mpiexec --partition=normal --nodes=1 --ntasks=2 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=1 --ntasks=3 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=1 --ntasks=4 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=1 --ntasks=5 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=1 --ntasks=6 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=1 --ntasks=7 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=1 --ntasks=8 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=2 --ntasks=9 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=2 --ntasks=10 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=2 --ntasks=11 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=2 --ntasks=12 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=2 --ntasks=13 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=2 --ntasks=14 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=2 --ntasks=15 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=2 --ntasks=16 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=3 --ntasks=17 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=3 --ntasks=18 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=3 --ntasks=19 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=3 --ntasks=20 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=3 --ntasks=21 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=3 --ntasks=22 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=3 --ntasks=23 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=3 --ntasks=24 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=4 --ntasks=25 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=4 --ntasks=26 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=4 --ntasks=27 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=4 --ntasks=28 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=4 --ntasks=29 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=4 --ntasks=30 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=4 --ntasks=31 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
mpiexec --partition=normal --nodes=4 --ntasks=32 --time=24:00:00 ../../../build/benchmarks/torsten_standard_comm ../../../test_data/ww_36_pmec_36.pm 1 ww_36_pmec_36 RMA
