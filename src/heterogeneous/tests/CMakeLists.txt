if(USE_CUDA)
    set_source_files_properties(
        test_gpu_alltoall.cpp
        test_gpu_alltoallv.cpp
        test_gpu_alltoall_init.cpp
        microbenchmarks.cpp
        simple_test.cpp
        allreduce_gpu.cpp
        allreduce_c2c.cpp
        allreduce_c2mc.cpp
        allreduce_c2mc_threaded.cpp
        allreduce_nccl.cpp
        allreduce_nccl_ipc.cpp
        allreduce_opt.cpp
        allreduce_nodeaware.cpp
        allreduce_mps.cpp
        allreduce_no_mps.cpp
        PROPERTIES LANGUAGE CUDA)
endif()

add_executable(test_gpu_alltoall test_gpu_alltoall.cpp)
target_link_libraries(test_gpu_alltoall mpi_advance gtest pthread)
add_test(GPUAlltoallTest ${GPUAwareMPIRUN} ./test_gpu_alltoall)

add_executable(test_gpu_alltoallv test_gpu_alltoallv.cpp)
target_link_libraries(test_gpu_alltoallv mpi_advance gtest pthread)
add_test(GPUAlltoallvTest ${GPUAwareMPIRUN} ./test_gpu_alltoallv)

add_executable(microbenchmarks microbenchmarks.cpp)
target_link_libraries(microbenchmarks mpi_advance gtest pthread)

add_executable(simple_test simple_test.cpp)
target_link_libraries(simple_test mpi_advance gtest pthread)

add_executable(allreduce_gpu allreduce_gpu.cpp)
target_link_libraries(allreduce_gpu mpi_advance gtest pthread)

add_executable(allreduce_c2c allreduce_c2c.cpp)
target_link_libraries(allreduce_c2c mpi_advance gtest pthread)

add_executable(allreduce_c2mc allreduce_c2mc.cpp)
target_link_libraries(allreduce_c2mc mpi_advance gtest pthread)

add_executable(allreduce_c2mc_threaded allreduce_c2mc_threaded.cpp)
target_link_libraries(allreduce_c2mc_threaded mpi_advance gtest pthread)

add_executable(allreduce_opt allreduce_opt.cpp)
target_link_libraries(allreduce_opt mpi_advance gtest pthread)

add_executable(allreduce_nodeaware allreduce_nodeaware.cpp)
target_link_libraries(allreduce_nodeaware mpi_advance gtest pthread)

add_executable(allreduce_mps allreduce_mps.cpp)
target_link_libraries(allreduce_mps mpi_advance gtest pthread)

add_executable(allreduce_no_mps allreduce_no_mps.cpp)
target_link_libraries(allreduce_no_mps mpi_advance gtest pthread)

add_executable(allreduce_nccl allreduce_nccl.cpp)
target_link_libraries(allreduce_nccl mpi_advance gtest pthread "/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/comm_libs/12.3/nccl/lib/libnccl.so.2")

add_executable(allreduce_nccl_ipc allreduce_nccl_ipc.cpp)
target_link_libraries(allreduce_nccl_ipc mpi_advance gtest pthread "/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/comm_libs/12.3/nccl/lib/libnccl.so.2")

#add_executable(test_gpu_alltoall_init test_gpu_alltoall_init.cpp)
#target_link_libraries(test_gpu_alltoall_init mpi_advance gtest pthread)
#add_test(GPUAlltoallInitTest ${} ./test_gpu_alltoall_init)
