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
#flux: --setattr=gpumode=TPX
#flux: --conf=resource.rediscover=true

export MPICH_GPU_SUPPORT_ENABLED=1
export HSA_XNACK=1
export HUGETLB_MORECORE=yes

cd $HOME/locality_develop/build_gpu/benchmarks

flux run -N 64 --verbose --setopt=mpibind=verbose --tasks-per-node=12 --gpus-per-node=12 ./gpu_allreduce


