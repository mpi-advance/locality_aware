add_executable(test_alltoall test_alltoall.cpp)
target_link_libraries(test_alltoall mpi_advance googletest pthread )
add_test(LocalityAlltoallTest mpirun -n 16 ./test_alltoall)

add_executable(test_alltoallv test_alltoallv.cpp)
target_link_libraries(test_alltoallv mpi_advance googletest pthread )
add_test(LocalityAlltoallvTest mpirun -n 16 ./test_alltoallv)

add_executable(test_allgather test_allgather.cpp)
target_link_libraries(test_allgather mpi_advance googletest pthread )
add_test(LocalityAllgatherTest mpirun -n 16 ./test_allgather)

