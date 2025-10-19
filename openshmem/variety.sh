#!/bin/bash
#SBATCH --partition=pdebug
#SBATCH --job-name=oshmem_a2av_sweep_elems
#SBATCH --time=00:40:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=openshmem_sweep.%j.out
#SBATCH --error=openshmem_sweep.%j.err


cd /g/g92/enamug/openshmem-alltoallv/openshmem/build

# OSHMEM/UCX setup
export OMPI_MCA_spml=ucx
export OMPI_MCA_pml=ucx
export SHMEM_SYMMETRIC_HEAP_SIZE=1G

MAP="--map-by ppr:8:node --bind-to core"
RESULTS_DIR="results_a2av_Ts_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Elements per destination (T values)
Tvals=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144)

iters=100
echo "Writing per-size outputs to: $RESULTS_DIR"

for T in "${Tvals[@]}"; do
  out="$RESULTS_DIR/bench_T${T}.out"
  err="$RESULTS_DIR/bench_T${T}.err"

  echo "=== Running: T=${T} elements per dest (bytes=$((T*8))), iters=${iters} ===" | tee -a "$out"

  oshrun $MAP -np 64 ./bench_alltoallv "$T" "$iters" >"$out" 2>"$err"
done

