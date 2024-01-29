include_directories(${mpi_advance_INCDIR})

if(USE_GPU)
    if(USE_CUDA)
        set_source_files_properties(
            gpu_alltoall.cpp
            alltoall_crs.cpp
            alltoallv_crs.cpp
            PROPERTIES LANGUAGE CUDA)
    endif()

add_executable(neighbor_collective_flip neighbor_collective_flip.cpp)
target_link_libraries(neighbor_collective_flip mpi_advance ${MPI_LIBRARIES})

