include_directories(${mpi_advance_INCDIR})
include_directories(${MPI_INCLUDE_PATH})

if(USE_GPU)
    if(USE_CUDA)
        set_source_files_properties(
            gpu_alltoall.cpp
            alltoall_crs.cpp
            alltoallv_crs.cpp
            microbenchmarks.cpp
            PROPERTIES LANGUAGE CUDA)
    endif()

    add_executable(gpu_alltoall gpu_alltoall.cpp)
    target_link_libraries(gpu_alltoall mpi_advance ${MPI_LIBRARIES} OpenMP::OpenMP_CXX)

endif()

add_executable(alltoall_crs alltoall_crs.cpp)
target_link_libraries(alltoall_crs mpi_advance ${MPI_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(alltoallv_crs alltoallv_crs.cpp)
target_link_libraries(alltoallv_crs mpi_advance ${MPI_LIBRARIES} OpenMP::OpenMP_CXX)

message(STATUS ${MPI_LIBRARIES} ${CMAKE_CUDA_FLAGS})
add_executable(microbenchmarks microbenchmarks.cpp)
target_link_libraries(microbenchmarks mpi_advance ${MPI_LIBRARIES})

