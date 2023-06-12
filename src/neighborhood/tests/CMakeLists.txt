if(USE_CUDA)
    set_source_files_properties(
        test_neighbor_alltoallv_init.cpp
        test_neighbor_alltoallw_init.cpp
        test_suitesparse_neighbor_alltoallv_init.cpp
        test_suitesparse_neighbor_alltoallw_init.cpp
        PROPERTIES LANGUAGE CUDA)
endif()

add_executable(test_neighbor_alltoallv_init test_neighbor_alltoallv_init.cpp)
target_link_libraries(test_neighbor_alltoallv_init mpi_advance gtest pthread )
add_test(PersistentNeighAlltoallvTest mpirun -n 16 ./test_neighbor_alltoallv_init)

add_executable(test_neighbor_alltoallw_init test_neighbor_alltoallw_init.cpp)
target_link_libraries(test_neighbor_alltoallw_init mpi_advance gtest pthread )
add_test(PersistentNeighAlltoallwTest mpirun -n 16 ./test_neighbor_alltoallw_init)

add_executable(test_suitesparse_neighbor_alltoallv_init 
    test_suitesparse_neighbor_alltoallv_init.cpp)
target_link_libraries(test_suitesparse_neighbor_alltoallv_init mpi_advance gtest pthread )
add_test(PersistentNeighAlltoallvSuitesparseTest 
    mpirun -n 16 ./test_suitesparse_neighbor_alltoallv_init)

add_executable(test_suitesparse_neighbor_alltoallw_init 
    test_suitesparse_neighbor_alltoallw_init.cpp)
target_link_libraries(test_suitesparse_neighbor_alltoallw_init mpi_advance gtest pthread )
add_test(PersistentNeighAlltoallwSuitesparseTest 
    mpirun -n 16 ./test_suitesparse_neighbor_alltoallw_init)

