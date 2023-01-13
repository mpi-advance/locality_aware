add_executable(test_nap_comm test_nap_comm.cpp)
target_link_libraries(test_nap_comm mpi_advance gtest pthread )
add_test(RandomCommTest mpirun -n 16 ./test_nap_comm)

add_executable(test_alltoallv test_alltoallv.cpp)
target_link_libraries(test_alltoallv mpi_advance gtest pthread )
add_test(RandomNeighTest mpirun -n 16 ./test_alltoallv)

