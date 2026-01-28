#!/bin/sh
#Submit using flux batch <filename>

#flux: --job-name=allreduce_N64
#flux: --output='allreduce_N64.{{id}}.out'
#flux: --error='allreduce_N64.{{id}}.err'
#flux: -N 64
#flux: -l # Add task rank prefixes to each line of output.
#flux: --setattr=thp=always # Enable Transparent Huge Pages (THP)
#flux: -t 20
#flux: -q pbatch # other available queues: pdebug
#flux: -x

export MPICH_GPU_SUPPORT_ENABLED=1
export HSA_XNACK=1

cd $HOME/locality_develop/build_gpu/benchmarks

flux run -N 64 --verbose --setopt=mpibind=verbose --tasks-per-node=84 ./allreduce


