// EXPECT_EQ and ASSERT_EQ are macros
// EXPECT_EQ test execution and continues even if there is a failure
// ASSERT_EQ test execution and aborts if there is a failure
// The ASSERT_* variants abort the program execution if an assertion fails
// while EXPECT_* variants continue with the run.


#include "gtest/gtest.h"
#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>

int main(int argc, char** argv)
{
#define LOCAL_COMM_PPN4
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    int temp=RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //


TEST(RandomCommTest, TestsInTests)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Test Integer Alltoall
    int max_i = 10;
    int max_s = pow(2, max_i);
    srand(time(NULL));
    std::vector<int> local_data(max_s*num_procs);
    std::vector<int> std_alltoall(max_s*num_procs);
    std::vector<int> pairwise_alltoall(max_s*num_procs);
    std::vector<int> device_data(max_s*num_procs);

    MPIX_Comm* xcomm;
    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);
    MPIX_Comm_device_init(xcomm);

    int n_gpus;
    gpuGetDeviceCount(&n_gpus);
    gpuSetDevice(xcomm->rank_gpu);

    int* local_data_d;
    int* alltoall_d;
    gpuMalloc((void**)&local_data_d, 
            max_s*num_procs*sizeof(int));
    gpuMalloc((void**)&alltoall_d, 
            max_s*num_procs*sizeof(int)); 

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);

        // Will only be clean for up to double digit process counts
        for (int i = 0; i < num_procs; i++)
            for (int j = 0; j < s; j++)
                local_data[i*s + j] = rank*10000 + i*100 + j;
        gpuMemcpy(local_data_d, 
                local_data.data(),
                s*num_procs*sizeof(int),
                gpuMemcpyHostToDevice);

        // Standard Alltoall
        PMPI_Alltoall(local_data.data(), 
                s,
                MPI_INT, 
                std_alltoall.data(), 
                s, 
                MPI_INT,
                MPI_COMM_WORLD);


        // Pairwise Alltoall
        alltoall_pairwise(local_data.data(), 
                s,
                MPI_INT, 
                pairwise_alltoall.data(), 
                s, 
                MPI_INT,
                xcomm);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_alltoall[j], pairwise_alltoall[j]);


	// Standard GPU Alltoall
        PMPI_Alltoall(local_data_d,
                s, 
                MPI_INT,
                alltoall_d,
                s,
                MPI_INT,
                MPI_COMM_WORLD);
        gpuMemcpy(device_data.data(), 
                alltoall_d, 
                s*num_procs*sizeof(int), 
                gpuMemcpyDeviceToHost);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_alltoall[j], device_data[j]);
        gpuMemset(alltoall_d, 0, s*num_procs*sizeof(int));

        // GPU-Aware Pairwise Alltoall
        gpu_aware_alltoall_pairwise(
                local_data_d, 
                s, 
                MPI_INT,
                alltoall_d,
                s, 
                MPI_INT,
                xcomm);
        gpuMemcpy(device_data.data(), 
                alltoall_d, 
                s*num_procs*sizeof(int), 
                gpuMemcpyDeviceToHost);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_alltoall[j], device_data[j]);
        gpuMemset(alltoall_d, 0, s*num_procs*sizeof(int));


        // GPU-Aware Nonblocking Alltoall
        gpu_aware_alltoall_nonblocking(
                local_data_d,
                s,
                MPI_INT,
                alltoall_d,
                s,
                MPI_INT,
                xcomm);
        gpuMemcpy(device_data.data(),
                alltoall_d,
                s*num_procs*sizeof(int),
                gpuMemcpyDeviceToHost);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_alltoall[j], device_data[j]);
        gpuMemset(alltoall_d, 0, s*num_procs*sizeof(int));


        // Copy-to-CPU Pairwise Alltoall
        copy_to_cpu_alltoall_pairwise(
                local_data_d, 
                s, 
                MPI_INT,
                alltoall_d,
                s, 
                MPI_INT,
                xcomm);
        gpuMemcpy(device_data.data(), 
                alltoall_d, 
                s*num_procs*sizeof(int), 
                gpuMemcpyDeviceToHost);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_alltoall[j], device_data[j]);
        gpuMemset(alltoall_d, 0, s*num_procs*sizeof(int));

        // Copy-to-CPU Nonblocking Alltoall
        copy_to_cpu_alltoall_nonblocking(
                local_data_d,
                s,
                MPI_INT,
                alltoall_d,
                s,
                MPI_INT,
                xcomm);
        gpuMemcpy(device_data.data(),
                alltoall_d,
                s*num_procs*sizeof(int),
                gpuMemcpyDeviceToHost);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_alltoall[j], device_data[j]);
        gpuMemset(alltoall_d, 0, s*num_procs*sizeof(int));
    }

    gpuFree(local_data_d);
    gpuFree(alltoall_d);

    MPIX_Comm_free(&xcomm);
}


