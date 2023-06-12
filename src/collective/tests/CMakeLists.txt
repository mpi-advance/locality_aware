if(USE_CUDA)
    set_source_files_properties(
        test_alltoall.cpp
        test_alltoallv.cpp
        test_suitesparse_alltoallv.cpp
        test_allgather.cpp
        PROPERTIES LANGUAGE CUDA)
endif()

add_executable(test_alltoall test_alltoall.cpp)
target_link_libraries(test_alltoall mpi_advance gtest pthread )
add_test(LocalityAlltoallTest mpirun -n 16 ./test_alltoall)

add_executable(test_alltoallv test_alltoallv.cpp)
target_link_libraries(test_alltoallv mpi_advance gtest pthread )
add_test(LocalityAlltoallvTest mpirun -n 16 ./test_alltoallv)

add_executable(test_suitesparse_alltoallv test_suitesparse_alltoallv.cpp)
target_link_libraries(test_suitesparse_alltoallv mpi_advance gtest pthread )
add_test(LocalitySuitesparseAlltoallvTest mpirun -n 16 ./test_suitesparse_alltoallv)

add_executable(test_allgather test_allgather.cpp)
target_link_libraries(test_allgather mpi_advance gtest pthread )
add_test(LocalityAllgatherTest mpirun -n 16 ./test_allgather)


