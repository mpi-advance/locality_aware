if(USE_CUDA)
    set_source_files_properties(
        test_neighbor_alltoallv_init.cpp
        test_neighbor_alltoallw_init.cpp
        test_suitesparse_neighbor_alltoallv_init.cpp
        test_suitesparse_neighbor_alltoallw_init.cpp
        test_suitesparse_alltoall_crs.cpp
        test_suitesparse_alltoallv_crs.cpp
        test_suitesparse_neighbor_reorder.cpp
        test_neighbor_reorder.cpp
        PROPERTIES LANGUAGE CUDA)
endif()

add_executable(test_neighbor_alltoallv_init test_neighbor_alltoallv_init.cpp)
target_link_libraries(test_neighbor_alltoallv_init mpi_advance gtest pthread )
add_test(PersistentNeighAlltoallvTest ${MPIRUN} -n 16 ./test_neighbor_alltoallv_init)

add_executable(test_neighbor_alltoallw_init test_neighbor_alltoallw_init.cpp)
target_link_libraries(test_neighbor_alltoallw_init mpi_advance gtest pthread )
add_test(PersistentNeighAlltoallwTest ${MPIRUN} -n 16 ./test_neighbor_alltoallw_init)

add_executable(test_suitesparse_neighbor_alltoallv_init 
    test_suitesparse_neighbor_alltoallv_init.cpp)
target_link_libraries(test_suitesparse_neighbor_alltoallv_init mpi_advance gtest pthread )
add_test(PersistentNeighAlltoallvSuitesparseTest 
    ${MPIRUN} -n 16 ./test_suitesparse_neighbor_alltoallv_init)

add_executable(test_suitesparse_neighbor_alltoallw_init 
    test_suitesparse_neighbor_alltoallw_init.cpp)
target_link_libraries(test_suitesparse_neighbor_alltoallw_init mpi_advance gtest pthread )
add_test(PersistentNeighAlltoallwSuitesparseTest 
    ${MPIRUN} -n 16 ./test_suitesparse_neighbor_alltoallw_init)

add_executable(test_partitioned_alltoallv 
    test_partitioned_alltoallv.cpp)
target_link_libraries(test_partitioned_alltoallv mpi_advance gtest pthread MPIPCL::MPIPCL)
add_test(PartitionedAlltoallvTest 
    ${MPIRUN} -n 16 ./test_partitioned_alltoallv)


add_executable(test_suitesparse_alltoall_crs test_suitesparse_alltoall_crs.cpp) 
target_link_libraries(test_suitesparse_alltoall_crs mpi_advance gtest pthread )
add_test(SuitesparseAlltoallCRS ${MPIRUN} -n 16 ./test_suitesparse_alltoall_crs)

add_executable(test_suitesparse_alltoallv_crs test_suitesparse_alltoallv_crs.cpp) 
target_link_libraries(test_suitesparse_alltoallv_crs mpi_advance gtest pthread )
add_test(SuitesparseAlltoallvCRS ${MPIRUN} -n 16 ./test_suitesparse_alltoallv_crs)


add_executable(test_neighbor_reorder test_neighbor_reorder.cpp)
target_link_libraries(test_neighbor_reorder mpi_advance gtest pthread )
add_test(PersistentNeighReorderTest ${MPIRUN} -n 16 ./test_neighbor_reorder)

add_executable(test_suitesparse_neighbor_reorder test_suitesparse_neighbor_reorder.cpp)
target_link_libraries(test_suitesparse_neighbor_reorder mpi_advance gtest pthread )
add_test(SuitesparseReorderTest ${MPIRUN} -n 16 ./test_suitesparse_neighbor_reorder)
