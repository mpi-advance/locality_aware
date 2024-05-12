add_executable(test_alltoall test_alltoall.cpp)
target_link_libraries(test_alltoall mpi_advance gtest pthread )
add_test(AlltoallTest mpirun -n 16 ./test_alltoall)

add_executable(test_alltoallv test_alltoallv.cpp)
target_link_libraries(test_alltoallv mpi_advance gtest pthread )
add_test(AlltoallvTest mpirun -n 16 ./test_alltoallv)

add_executable(test_suitesparse_alltoallv test_suitesparse_alltoallv.cpp)
target_link_libraries(test_suitesparse_alltoallv mpi_advance gtest pthread )
add_test(SuitesparseAlltoallvTest mpirun -n 16 ./test_suitesparse_alltoallv)

add_executable(test_allgather test_allgather.cpp)
target_link_libraries(test_allgather mpi_advance gtest pthread )
add_test(AllgatherTest mpirun -n 16 ./test_allgather)


add_executable(test_alltoall_init test_alltoall_init.cpp)
target_link_libraries(test_alltoall_init mpi_advance gtest pthread )
add_test(PersistentAlltoallTest mpirun -n 16 ./test_alltoall)

add_executable(test_alltoallv_init test_alltoallv_init.cpp)
target_link_libraries(test_alltoallv_init mpi_advance gtest pthread )
add_test(PersistentAlltoallvTest mpirun -n 16 ./test_alltoallv)
