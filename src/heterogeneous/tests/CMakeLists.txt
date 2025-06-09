if(USE_CUDA)
    set_source_files_properties(
        test_gpu_alltoall.cpp
        test_gpu_alltoallv.cpp
        PROPERTIES LANGUAGE CUDA)
endif()

add_executable(test_gpu_alltoall test_gpu_alltoall.cpp)
target_link_libraries(test_gpu_alltoall mpi_advance)
add_test(GPUAlltoallTest ${GPUAwareMPIRUN} ./test_gpu_alltoall)

add_executable(test_gpu_alltoallv test_gpu_alltoallv.cpp)
target_link_libraries(test_gpu_alltoallv mpi_advance)
add_test(GPUAlltoallvTest ${GPUAwareMPIRUN} ./test_gpu_alltoallv)

