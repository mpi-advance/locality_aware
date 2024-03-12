include_directories(${mpi_advance_INCDIR})

add_executable(alltoall_crs alltoall_crs.cpp)
target_link_libraries(alltoall_crs mpi_advance ${MPI_LIBRARIES})

add_executable(alltoallv_crs alltoallv_crs.cpp)
target_link_libraries(alltoallv_crs mpi_advance ${MPI_LIBRARIES})

