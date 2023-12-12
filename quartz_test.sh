#!/bin/bash
set -e

#export MPI_NAME=openmpi
export MPI_NAME=mvapich
#export BUILD_FOLDER=./build
export BUILD_FOLDER=./build-${MPI_NAME}
export EXECUTABLE=${BUILD_FOLDER}/benchmarks/neighbor_collective
export EXPERIMENT=6

mkdir -p data/${EXPERIMENT}/${MPI_NAME}

# Iterate over job sizes (node count)
#for i in {1,2,4,8,16,32,64,128,256,512,1024}
#for i in {1,2}
for i in {1,1024}
do
    export NUM_NODES=$i
    export JOB_NAME=MPIX_TOPO_${NUM_NODES}
    export MODULES_TO_USE="gcc/8.5 mvapich2/2.3.7"
    #export MODULES_TO_USE="gcc/11.2.1 openmpi"
    #export MODULES_TO_USE="openmpi"
    
    echo "Running ${JOB_NAME} on ${NUM_NODES} nodes using ${MODULES_TO_USE}."

    echo "#!/bin/bash" >> temp_sbatch
    echo "#SBATCH --job-name=${JOB_NAME}" >> temp_sbatch
    echo "#SBATCH --nodes=${NUM_NODES}" >> temp_sbatch
    echo "#SBATCH --tasks-per-node=36" >> temp_sbatch
    echo "#SBATCH --cpus-per-task=1" >> temp_sbatch
    echo "#SBATCH --time=1:00:00" >> temp_sbatch
    echo "#SBATCH --sockets-per-node=2" >> temp_sbatch
    echo "#SBATCH --cores-per-socket=18" >> temp_sbatch
    echo "#SBATCH --partition=pbatch" >> temp_sbatch
    echo "#SBATCH --output=data/${EXPERIMENT}/${MPI_NAME}/output-%A.out" >> temp_sbatch
    
    #echo "module load ${MODULES_TO_USE}" >> temp_sbatch
    echo "echo \"Number of Nodes: $NUM_NODES\"" >> temp_sbatch 	
    echo "echo \"Modules: $MODULES_TO_USE\"" >> temp_sbatch 	

    echo "echo \"Starting Chebyshev4\"" >> temp_sbatch
    echo "srun ${EXECUTABLE} ./test_data/Chebyshev4.pm " >> temp_sbatch
    echo "echo \"Starting Wordnet3\"" >> temp_sbatch
    echo "srun ${EXECUTABLE} ./test_data/Wordnet3.pm " >> temp_sbatch
    echo "echo \"Starting water_tank\"" >> temp_sbatch
    echo "srun ${EXECUTABLE} ./test_data/water_tank.pm " >> temp_sbatch
    echo "echo \"Starting mycielskian17\"" >> temp_sbatch
    echo "srun ${EXECUTABLE} ./test_data/mycielskian17.pm " >> temp_sbatch
    
    # Launch generated script
    sbatch temp_sbatch
    sbatch temp_sbatch
    sbatch temp_sbatch

    # Remove generated script
    rm temp_sbatch
done
