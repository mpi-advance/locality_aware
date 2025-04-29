if(USE_CUDA)
    set_source_files_properties(
        test_alltoall.cpp
        test_alltoallv.cpp
        test_suitesparse_alltoallv.cpp
        PROPERTIES LANGUAGE CUDA)
endif()

add_executable(test_alltoall test_alltoall.cpp)
target_link_libraries(test_alltoall mpi_advance GTest::gtest GTest::gtest_main pthread )
add_test(LocalityAlltoallTest ${MPIRUN} -n 16 ./test_alltoall)

add_executable(test_alltoallv test_alltoallv.cpp)
target_link_libraries(test_alltoallv mpi_advance GTest::gtest GTest::gtest_main pthread )
add_test(LocalityAlltoallvTest ${MPIRUN} -n 16 ./test_alltoallv)

add_executable(test_suitesparse_alltoallv test_suitesparse_alltoallv.cpp)
target_link_libraries(test_suitesparse_alltoallv mpi_advance GTest::gtest GTest::gtest_main pthread )
add_test(LocalitySuitesparseAlltoallvTest ${MPIRUN} -n 16 ./test_suitesparse_alltoallv)
