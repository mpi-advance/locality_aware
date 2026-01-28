#!/bin/sh
#Submit using flux batch <filename>

#flux: --job-name=allreduce_N32
#flux: --output='allreduce_N32.{{id}}.out'
#flux: --error='allreduce_N32.{{id}}.err'
#flux: -N 32
#flux: -l # Add task rank prefixes to each line of output.
#flux: --setattr=thp=always # Enable Transparent Huge Pages (THP)
#flux: -t 20
#flux: -q pbatch # other available queues: pdebug
#flux: -x
#flux: --conf=resource.rediscover=true

export MPICH_GPU_SUPPORT_ENABLED=1
export HSA_XNACK=1

cd $HOME/locality_develop/build_gpu/benchmarks

flux run -N 32 --verbose --tasks-per-node=4 --gpus-per-node=4 ./gpu_allreduce



