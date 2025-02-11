#!/bin/bash
#BSUB -J rma_alltoallv_n4_ppn4
#BSUB -e rma_alltoallv_n4_ppn4.%J.err
#BSUB -o rma_alltoall4_n4_ppn4.%J.out
#BSUB -nnodes 18
##BSUB -q pbatch
#BSUB -q pdebug
#BSUB -W 00:15

cd /g/g92/enamug/clean/GPU_locality_aware/locality_aware/build/benchmarks

jsrun -a4 -c4 -r1 -n4 --latency_priority=cpu-cpu --launch_distribution=packed --print_placement=1 ./alltoallv_rma

#jsrun -a44 -c44 -r1 --latency_priority=cpu-cpu --launch_distribution=packed --print_placement=1 ./alltoallv_rma

