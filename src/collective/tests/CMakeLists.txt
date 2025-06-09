add_executable(test_alltoall test_alltoall.cpp)
target_link_libraries(test_alltoall mpi_advance)
add_test(AlltoallTest ${MPIRUN} -n 16 ./test_alltoall)

add_executable(test_alltoallv test_alltoallv.cpp)
target_link_libraries(test_alltoallv mpi_advance)
add_test(AlltoallvTest ${MPIRUN} -n 16 ./test_alltoallv)

add_executable(test_suitesparse_alltoallv test_suitesparse_alltoallv.cpp)
target_link_libraries(test_suitesparse_alltoallv mpi_advance)
add_test(SuitesparseAlltoallvTest ${MPIRUN} -n 16 ./test_suitesparse_alltoallv)
