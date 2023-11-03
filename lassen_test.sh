#!/bin/bash
set -e

#export BUILD_FOLDER=./build
export BUILD_FOLDER=./build-spectrum
export EXECUTABLE=${BUILD_FOLDER}/benchmarks/neighbor_collective
export EXPERIMENT=lassen/6

mkdir -p data/${EXPERIMENT}

# Iterate over job sizes (node count)
#for i in {1,2,4,8,16,32,64,128,256}
for i in {32,64,128,256}
do
    export NUM_NODES=$i
    export JOB_NAME=MPIX_TOPO_${NUM_NODES}
    #export MODULES_TO_USE="gcc/8.5 mvapich2/2.3.7"
    #export MODULES_TO_USE="gcc/11.2.1 openmpi"
    #export MODULES_TO_USE="openmpi"
    export MODULES_TO_USE="gcc/8.3.1"

    echo "Running ${JOB_NAME} on ${NUM_NODES} nodes using ${MODULES_TO_USE}."

    echo "#!/bin/bash" >> temp_sbatch
    echo "#BSUB -J ${JOB_NAME}" >> temp_sbatch
    echo "#BSUB -nnodes ${NUM_NODES}" >> temp_sbatch
    echo "#BSUB -W 60" >> temp_sbatch
    echo "#BSUB -q pbatch" >> temp_sbatch
    echo "#BSUB -o data/${EXPERIMENT}/output-${i}.out" >> temp_sbatch
    
    echo "module load ${MODULES_TO_USE}" >> temp_sbatch
    echo "echo \"Number of Nodes: $NUM_NODES\"" >> temp_sbatch 	
    echo "echo \"Modules: $MODULES_TO_USE\"" >> temp_sbatch 	

    echo "echo \"Starting Chebyshev4\"" >> temp_sbatch
    echo "lrun -N${i} -T40 ${EXECUTABLE} ./test_data/Chebyshev4.pm " >> temp_sbatch
    #echo "echo \"Starting Wordnet3\"" >> temp_sbatch
    #echo "lrun -N${i} -T40 ${EXECUTABLE} ./test_data/Wordnet3.pm " >> temp_sbatch
    echo "echo \"Starting water_tank\"" >> temp_sbatch
    echo "lrun -N${i} -T40 ${EXECUTABLE} ./test_data/water_tank.pm " >> temp_sbatch
    echo "echo \"Starting mycielskian17\"" >> temp_sbatch
    echo "lrun -N${i} -T40 ${EXECUTABLE} ./test_data/mycielskian17.pm " >> temp_sbatch
    
    # Launch generated script
    bsub temp_sbatch

    # Remove generated script
    rm temp_sbatch
done
