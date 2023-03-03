module load gcc
module load openmpi
module load cmake

rm -rf build
cd build/
cmake .. ./
make 
cd benchmarks/
make
