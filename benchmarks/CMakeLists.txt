include_directories(${mpi_advance_INCDIR})

add_executable(neighbor_collective neighbor_collective.cpp)
target_link_libraries(neighbor_collective mpi_advance ${MPI_LIBRARIES})

add_executable(neighbor_collective_flip neighbor_collective_flip.cpp)
target_link_libraries(neighbor_collective_flip mpi_advance ${MPI_LIBRARIES})

