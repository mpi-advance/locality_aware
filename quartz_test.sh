#!/bin/bash
set -e

# Iterate over job sizes (node count)
for i in {1,2} #,4,8,16}
do
    export NUM_NODES=$i
    export NUM_PROC_PER_NODE=36
    export JOB_NAME=MPIX_TOPO_${NUM_NODES}
    
    echo "Running ${JOB_NAME} on ${NUM_NODES} nodes."

    echo "#!/bin/bash" >> temp_sbatch
    echo "#SBATCH --job-name=${JOB_NAME}" >> temp_sbatch
    echo "#SBATCH --nodes=${NUM_NODES}" >> temp_sbatch
    echo "#SBATCH --tasks-per-node=36" >> temp_sbatch
    echo "#SBATCH --cpus-per-task=1" >> temp_sbatch
    echo "#SBATCH --time=1:00:00" >> temp_sbatch
    echo "#SBATCH --sockets-per-node=2" >> temp_sbatch
    echo "#SBATCH --cores-per-socket=18" >> temp_sbatch
    echo "#SBATCH --partition=pbatch" >> temp_sbatch
    echo "module load gcc openmpi" >> temp_sbatch
    
    echo "echo \"Starting bcsstk\"" >> temp_sbatch
    echo "srun ./build/benchmarks/neighbor_collective ./test_data/bcsstk01.pm " >> temp_sbatch
    echo "echo \"Starting bidb_49_3\"" >> temp_sbatch
    echo "srun ./build/benchmarks/neighbor_collective ./test_data/bibd_49_3.pm " >> temp_sbatch
    echo "echo \"Starting radfr1\"" >> temp_sbatch
    echo "srun ./build/benchmarks/neighbor_collective ./test_data/radfr1.pm " >> temp_sbatch
    echo "echo \"Starting odepa400\"" >> temp_sbatch
    echo "srun ./build/benchmarks/neighbor_collective ./test_data/odepa400.pm " >> temp_sbatch
    echo "echo \"Starting SmaGri\"" >> temp_sbatch
    echo "srun ./build/benchmarks/neighbor_collective ./test_data/SmaGri.pm " >> temp_sbatch

    # Launch generated script
    sbatch temp_sbatch

    # Remove generated script
    rm temp_sbatch
done
