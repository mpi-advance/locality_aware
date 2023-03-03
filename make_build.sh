module load gcc
module load openmpi
module load cmake

rm -rf build
mkdir build
cd build/
cmake .. ./
make 
cd benchmarks/
make
