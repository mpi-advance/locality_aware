add_executable(test_alltoall test_alltoall.cpp)
target_link_libraries(test_alltoall mpi_advance googletest pthread )
add_test(RandomNeighTest mpirun -n 16 ./test_alltoall)

