include_directories(${mpi_advance_INCDIR})

add_executable(bruck_allgather bruck_allgather.cpp)
target_link_libraries(bruck_allgather mpi_advance ${MPI_LIBRARIES})

