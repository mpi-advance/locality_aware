if(USE_CUDA)
    set_source_files_properties(
        test_gpu_alltoall.cpp
        PROPERTIES LANGUAGE CUDA)
endif()

add_executable(test_gpu_alltoall test_gpu_alltoall.cpp)
target_link_libraries(test_gpu_alltoall mpi_advance gtest pthread)
add_test(GPUAlltoallTest mpirun -n 16 ./test_gpu_alltoall)


