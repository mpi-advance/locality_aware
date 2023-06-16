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
#include <numeric>
#include <set>

#include "tests/sparse_mat.hpp"
#include "tests/par_binary_IO.hpp"

void test_matrix(const char* filename)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    MPIX_Comm* xcomm;
    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);

    // Read suitesparse matrix
    ParMat<int> A;
    readParMatrix(filename, A);
    form_comm(A);

    std::vector<int> send_vals(A.on_proc.n_rows);
    std::iota(send_vals.begin(), send_vals.end(), 0);
    for (int i = 0; i < A.on_proc.n_rows; i++)
        send_vals[i] += (rank*1000);

    // Alltoallv_send_vals must be ordered (dest 0 to num_procs-1)
    std::vector<int> proc_pos(num_procs, -1);
    for (int i = 0; i < A.send_comm.n_msgs; i++)
        proc_pos[A.send_comm.procs[i]] = i;

    std::vector<int> alltoallv_send_vals(A.send_comm.size_msgs);
    int start, end, idx;
    int ctr = 0;
    for (int i = 0; i < num_procs; i++)
    {
        idx = proc_pos[i];
        if (proc_pos[i] < 0) continue;

        start = A.send_comm.ptr[idx];
        end = A.send_comm.ptr[idx+1];
        for (int j = start; j < end; j++)
        {
            alltoallv_send_vals[ctr++] = send_vals[A.send_comm.idx[j]];
        }
    }

    std::vector<int> sendcounts(num_procs, 0); 
    std::vector<int> sdispls(num_procs+1);
    std::vector<int> recvcounts(num_procs, 0);
    std::vector<int> rdispls(num_procs+1);

    for (int i = 0; i < A.send_comm.n_msgs; i++)
        sendcounts[A.send_comm.procs[i]] = A.send_comm.ptr[i+1] - A.send_comm.ptr[i];
    for (int i = 0; i < A.recv_comm.n_msgs; i++)
        recvcounts[A.recv_comm.procs[i]] = A.recv_comm.ptr[i+1] - A.recv_comm.ptr[i];

    sdispls[0] = 0;
    rdispls[0] = 0;
    for (int i = 0; i < num_procs; i++)
    {
        sdispls[i+1] = sdispls[i] + sendcounts[i];
        rdispls[i+1] = rdispls[i] + recvcounts[i];
    }

    int n_gpus;
    gpuGetDeviceCount(&n_gpus);
    gpuSetDevice(xcomm->rank_gpu);

    int *sendbuf_d, *recvbuf_d;
    gpuMalloc((void**)&sendbuf_d,A.send_comm.size_msgs*sizeof(int));
    gpuMalloc((void**)&recvbuf_d,A.recv_comm.size_msgs*sizeof(int));
    gpuMemcpy(sendbuf_d, alltoallv_send_vals.data(), A.send_comm.size_msgs*sizeof(int),
            gpuMemcpyHostToDevice);

    std::vector<int> pmpi_recv_vals(A.recv_comm.size_msgs);
    std::vector<int> gpu_recv_vals(A.recv_comm.size_msgs);

    // Inter-CPU Alltoallv
    PMPI_Alltoallv(alltoallv_send_vals.data(), 
            sendcounts.data(),
            sdispls.data(),
            MPI_INT,
            pmpi_recv_vals.data(),
            recvcounts.data(),
            rdispls.data(),
            MPI_INT,
            xcomm->global_comm);

    // Inter-GPU Alltoallv
    PMPI_Alltoallv(sendbuf_d, 
            sendcounts.data(),
            sdispls.data(),
            MPI_INT, 
            recvbuf_d,
            recvcounts.data(),
            rdispls.data(),
            MPI_INT,
            xcomm->global_comm);
    gpuMemcpy(gpu_recv_vals.data(), recvbuf_d, A.recv_comm.size_msgs*sizeof(int), gpuMemcpyDeviceToHost);
    for (int i = 0; i < A.recv_comm.size_msgs; i++)
    {
        ASSERT_EQ(pmpi_recv_vals[i], gpu_recv_vals[i]);
    }
    cudaMemset(recvbuf_d, 0, A.recv_comm.size_msgs*sizeof(int));

    gpu_aware_alltoallv_pairwise(sendbuf_d,
            sendcounts.data(),
            sdispls.data(),
            MPI_INT,
            recvbuf_d,
            recvcounts.data(),
            rdispls.data(),
            MPI_INT,
            xcomm);
    gpuMemcpy(gpu_recv_vals.data(), recvbuf_d, A.recv_comm.size_msgs*sizeof(int), gpuMemcpyDeviceToHost);
    for (int i = 0; i < A.recv_comm.size_msgs; i++)
    {
        ASSERT_EQ(pmpi_recv_vals[i], gpu_recv_vals[i]);
    }
    cudaMemset(recvbuf_d, 0, A.recv_comm.size_msgs*sizeof(int));

    gpu_aware_alltoallv_nonblocking(sendbuf_d,
            sendcounts.data(),
            sdispls.data(),
            MPI_INT,
            recvbuf_d,
            recvcounts.data(),
            rdispls.data(),
            MPI_INT,
            xcomm);
    gpuMemcpy(gpu_recv_vals.data(), recvbuf_d, A.recv_comm.size_msgs*sizeof(int), gpuMemcpyDeviceToHost);
    for (int i = 0; i < A.recv_comm.size_msgs; i++)
    {
        ASSERT_EQ(pmpi_recv_vals[i], gpu_recv_vals[i]);
    }
    cudaMemset(recvbuf_d, 0, A.recv_comm.size_msgs*sizeof(int));

    copy_to_cpu_alltoallv_pairwise(sendbuf_d,
            sendcounts.data(),
            sdispls.data(),
            MPI_INT,
            recvbuf_d,
            recvcounts.data(),
            rdispls.data(),
            MPI_INT,
            xcomm);
    gpuMemcpy(gpu_recv_vals.data(), recvbuf_d, A.recv_comm.size_msgs*sizeof(int), gpuMemcpyDeviceToHost);
    for (int i = 0; i < A.recv_comm.size_msgs; i++)
    {
        ASSERT_EQ(pmpi_recv_vals[i], gpu_recv_vals[i]);
    }
    cudaMemset(recvbuf_d, 0, A.recv_comm.size_msgs*sizeof(int));

    copy_to_cpu_alltoallv_nonblocking(sendbuf_d,
            sendcounts.data(),
            sdispls.data(),
            MPI_INT,
            recvbuf_d,
            recvcounts.data(),
            rdispls.data(),
            MPI_INT,
            xcomm);
    gpuMemcpy(gpu_recv_vals.data(), recvbuf_d, A.recv_comm.size_msgs*sizeof(int), gpuMemcpyDeviceToHost);
    for (int i = 0; i < A.recv_comm.size_msgs; i++)
    {
        ASSERT_EQ(pmpi_recv_vals[i], gpu_recv_vals[i]);
    }
    cudaMemset(recvbuf_d, 0, A.recv_comm.size_msgs*sizeof(int));

    threaded_alltoallv_pairwise(sendbuf_d,
            sendcounts.data(),
            sdispls.data(),
            MPI_INT,
            recvbuf_d,
            recvcounts.data(),
            rdispls.data(),
            MPI_INT,
            xcomm);
    gpuMemcpy(gpu_recv_vals.data(), recvbuf_d, A.recv_comm.size_msgs*sizeof(int), gpuMemcpyDeviceToHost);
    for (int i = 0; i < A.recv_comm.size_msgs; i++)
    {
        ASSERT_EQ(pmpi_recv_vals[i], gpu_recv_vals[i]);
    }
    cudaMemset(recvbuf_d, 0, A.recv_comm.size_msgs*sizeof(int));

    threaded_alltoallv_nonblocking(sendbuf_d,
            sendcounts.data(),
            sdispls.data(),
            MPI_INT,
            recvbuf_d,
            recvcounts.data(),
            rdispls.data(),
            MPI_INT,
            xcomm);
    gpuMemcpy(gpu_recv_vals.data(), recvbuf_d, A.recv_comm.size_msgs*sizeof(int), gpuMemcpyDeviceToHost);
    for (int i = 0; i < A.recv_comm.size_msgs; i++)
    {
        ASSERT_EQ(pmpi_recv_vals[i], gpu_recv_vals[i]);
    }
    cudaMemset(recvbuf_d, 0, A.recv_comm.size_msgs*sizeof(int));

    gpuFree(sendbuf_d);
    gpuFree(recvbuf_d);

    MPIX_Comm_free(xcomm);
}

int main(int argc, char** argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    int temp=RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //


TEST(RandomCommTest, TestsInTests)
{
    // Get MPI Information
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    test_matrix("../../../../test_data/dwt_162.pm");
    test_matrix("../../../../test_data/odepa400.pm");
    test_matrix("../../../../test_data/ww_36_pmec_36.pm");
}

