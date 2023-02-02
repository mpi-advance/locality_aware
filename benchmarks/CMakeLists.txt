include_directories(${mpi_advance_INCDIR})

add_executable(bruck_allgather bruck_allgather.cpp)
target_link_libraries(bruck_allgather mpi_advance ${MPI_LIBRARIES})

add_executable(p2p_alltoall p2p_alltoall.cpp)
target_link_libraries(p2p_alltoall mpi_advance ${MPI_LIBRARIES})

add_executable(p2p_alltoallv p2p_alltoallv.cpp)
target_link_libraries(p2p_alltoallv mpi_advance ${MPI_LIBRARIES})

add_executable(torsten_standard_comm torsten_standard_comm.cpp)
target_link_libraries(torsten_standard_comm mpi_advance ${MPI_LIBRARIES})
