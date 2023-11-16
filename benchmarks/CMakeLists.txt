include_directories(${mpi_advance_INCDIR})

if(USE_CUDA)
    set_source_files_properties(
        gpu_alltoall.cpp
        gpu_threaded_alltoall.cpp
        PROPERTIES LANGUAGE CUDA)

    add_executable(gpu_alltoall gpu_alltoall.cpp)
    target_link_libraries(gpu_alltoall mpi_advance ${MPI_LIBRARIES} OpenMP::OpenMP_CXX)

    add_executable(gpu_threaded_alltoall gpu_threaded_alltoall.cpp)
    target_compile_options(gpu_threaded_alltoall PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: # Flags for CUDA Objects
    -lineinfo
    -Xcompiler -fopenmp
    >
    )
    target_link_libraries(gpu_threaded_alltoall mpi_advance ${MPI_LIBRARIES} OpenMP::OpenMP_CXX)

else()
    add_executable(bruck_allgather bruck_allgather.cpp)
    target_link_libraries(bruck_allgather mpi_advance ${MPI_LIBRARIES})

    add_executable(p2p_alltoall p2p_alltoall.cpp)
    target_link_libraries(p2p_alltoall mpi_advance ${MPI_LIBRARIES})

    add_executable(p2p_alltoallv p2p_alltoallv.cpp)
    target_link_libraries(p2p_alltoallv mpi_advance ${MPI_LIBRARIES})

    add_executable(test_comm_setup, test_comm_setup.cpp)
    target_link_libraries(test_comm_setup mpi_advance ${MPI_LIBRARIES})

    add_executable(test_comm_setup_GPU test_comm_setup_GPU.cpp)
    target_link_libraries(test_comm_setup_GPU mpi_advance ${MPI_LIBRARIES})

    add_executable(dynamic_p2p dynamic_p2p.cpp)
    target_link_libraries(dynamic_p2p mpi_advance ${MPI_LIBRARIES})
endif()
