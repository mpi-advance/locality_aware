include_directories(${mpi_advance_INCDIR})

if(USE_GPU)
    if(USE_CUDA)
        set_source_files_properties(
            gpu_alltoall.cpp
            gpu_threaded_alltoall.cpp
            gpu_extraproc_alltoall.cpp
  			threaded_p2p.cpp
            no_thread_launches.cpp
            PROPERTIES LANGUAGE CUDA)
    endif()

    add_executable(gpu_alltoall gpu_alltoall.cpp)
    target_link_libraries(gpu_alltoall mpi_advance ${MPI_LIBRARIES} OpenMP::OpenMP_CXX)

    add_executable(gpu_threaded_alltoall gpu_threaded_alltoall.cpp)
    if (USE_CUDA)
        target_compile_options(gpu_threaded_alltoall PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>: # Flags for CUDA Objects
        -lineinfo
        -Xcompiler -fopenmp
        >
        )
    endif()
    target_link_libraries(gpu_threaded_alltoall mpi_advance ${MPI_LIBRARIES} OpenMP::OpenMP_CXX)
    add_executable(no_thread_launches no_thread_launches.cpp)
    if (USE_CUDA)
        target_compile_options(no_thread_launches PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>: # Flags for CUDA Objects
        -lineinfo
        -Xcompiler -fopenmp
        >
        )
    endif()
    target_link_libraries(no_thread_launches mpi_advance ${MPI_LIBRARIES} OpenMP::OpenMP_CXX)

    add_executable(gpu_extraproc_alltoall gpu_extraproc_alltoall.cpp)
    #if (USE_CUDA)
    #    target_compile_options(gpu_extraproc_alltoall PRIVATE
    #    $<$<COMPILE_LANGUAGE:CUDA>: # Flags for CUDA Objects
    #    -lineinfo
    #    -Xcompiler -fopenmp
    #    >
    #    )
    #endif()
    #target_link_libraries(gpu_extraproc_alltoall mpi_advance ${MPI_LIBRARIES} OpenMP::OpenMP_CXX)
    target_link_libraries(gpu_extraproc_alltoall mpi_advance ${MPI_LIBRARIES})

    add_executable(threaded_p2p threaded_p2p.cpp)
    target_link_libraries(threaded_p2p mpi_advance ${MPI_LIBRARIES} OpenMP::OpenMP_CXX)


else()
    add_executable(bruck_allgather bruck_allgather.cpp)
    target_link_libraries(bruck_allgather mpi_advance ${MPI_LIBRARIES})

    add_executable(p2p_alltoall p2p_alltoall.cpp)
    target_link_libraries(p2p_alltoall mpi_advance ${MPI_LIBRARIES})

    add_executable(p2p_alltoallv p2p_alltoallv.cpp)
    target_link_libraries(p2p_alltoallv mpi_advance ${MPI_LIBRARIES})
endif()
