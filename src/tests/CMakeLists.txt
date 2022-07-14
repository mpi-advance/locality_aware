add_executable(test_alltoallv test_alltoallv.cpp)
target_link_libraries(test_alltoallv mpi_advance googletest pthread )
add_test(RandomNeighTest mpirun -n 16 ./test_alltoallv)

