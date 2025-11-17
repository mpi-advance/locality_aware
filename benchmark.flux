#!/bin/sh
#Submit using flux batch <filename>

#flux: --job-name=allreduce_N2
#flux: --output='allreduce_N2.{{id}}.out'
#flux: --error='allreduce_N2.{{id}}.err'
#flux: -N 2
#flux: -l # Add task rank prefixes to each line of output.
#flux: --setattr=thp=always # Enable Transparent Huge Pages (THP)
#flux: -t 20
#flux: -q pbatch # other available queues: pdebug

cd $HOME/locality_develop/build/benchmarks

flux_run -N 2 --tasks-per-node=64 ./allreduce


