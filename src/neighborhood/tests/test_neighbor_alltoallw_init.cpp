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

#include "neighbor_data.h"



int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
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

    setenv("PPN", "4", 1);

    // Initial communication info (standard)
    int local_size = 10000; // Number of variables each rank stores
    MPIX_Data<MPI_Aint> send_data;
    MPIX_Data<MPI_Aint> recv_data;
    form_initial_communicator(local_size, &send_data, &recv_data);

    // Test correctness of communication
    std::vector<int> std_recv_vals(recv_data.size_msgs);
    std::vector<int> new_recv_vals(recv_data.size_msgs);
    std::vector<int> send_vals(local_size);
    int val = local_size*rank;
    for (int i = 0; i < local_size; i++)
    {
        send_vals[i] = val++;
    }

    std::vector<int> alltoallv_send_vals(send_data.size_msgs);
    for (int i = 0; i < send_data.size_msgs; i++)
        alltoallv_send_vals[i] = send_vals[send_data.indices[i]];

    MPI_Comm std_comm;
    MPI_Status status;
    MPIX_Comm* neighbor_comm;
    MPIX_Request* neighbor_request;
    std::vector<MPI_Datatype> sendtypes(num_procs, MPI_INT);
    std::vector<MPI_Datatype> recvtypes(num_procs, MPI_INT);

    MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD,
            recv_data.num_msgs, 
            recv_data.procs.data(), 
            recv_data.counts.data(),
            send_data.num_msgs, 
            send_data.procs.data(),
            send_data.counts.data(),
            MPI_INFO_NULL, 
            0, 
            &std_comm);

    /*printf("Rank %d sending %d msgs\n", rank, send_data.num_msgs);
    for (int i = 0; i < send_data.num_msgs; i++)
    {
        printf("Rank %d sending %d (from %ld to %ld) to %d\n",
                rank, send_data.counts[i],
                send_data.indptr[i], send_data.indptr[i+1],
                send_data.procs[i]);
        for (MPI_Aint j = send_data.indptr[i]; j < send_data.indptr[i+1]; j++)
        {
            printf("Rank %d sending val[%ld] = %d\n", rank, j, alltoallv_send_vals[j]);
        }
    } */

    if (rank == 0) 
    {
        printf("Rank 0 send:");
        for (int i = 0; i < send_data.num_msgs; i++)
            printf("Counts[%d] = %d, ", i, send_data.counts[i]);
        for (int i = 0; i < send_data.num_msgs; i++)
        {
            printf("Displs[%d] = %ld, ", i, send_data.indptr[i]);
            for (int j = send_data.indptr[i]; j < send_data.indptr[i] + send_data.counts[i]; j++)
                printf("sendbuf[%d] = %d, ", j, alltoallv_send_vals[j]);
        }
        printf("\n");
    }


    if (rank == 1) 
    {
        printf("Rank 1 send:");
        for (int i = 0; i < send_data.num_msgs; i++)
            printf("Counts[%d] = %d, ", i, send_data.counts[i]);
        for (int i = 0; i < send_data.num_msgs; i++)
        {
            printf("Displs[%d] = %ld, ", i, send_data.indptr[i]);
            for (int j = send_data.indptr[i]; j < send_data.indptr[i] + send_data.counts[i]; j++)
                printf("sendbuf[%d] = %d, ", j, alltoallv_send_vals[j]);
        }
        printf("\n");
    }




    MPI_Neighbor_alltoallw(alltoallv_send_vals.data(), 
            send_data.counts.data(),
            send_data.indptr.data(), 
            sendtypes.data(),
            std_recv_vals.data(), 
            recv_data.counts.data(),
            recv_data.indptr.data(), 
            recvtypes.data(),
            std_comm);


    /*printf("Rank %d recvd %d msgs\n", rank, recv_data.num_msgs);
    for (int i = 0; i < recv_data.num_msgs; i++)
    {
        printf("Rank %d recvd %d (from %ld to %ld) from %d\n",
                rank, recv_data.counts[i],
                recv_data.indptr[i], recv_data.indptr[i+1],
                recv_data.procs[i]);
        for (MPI_Aint j = recv_data.indptr[i]; j < recv_data.indptr[i+1]; j++)
        {
            printf("Rank %d recvd val[%ld] = %d\n", rank, j, std_recv_vals[j]);
        }
    } */
    if (rank == 3)
    {
        printf("Rank 3 recv: ");
        for (int i = 0; i < recv_data.num_msgs; i++)
            printf("Counts[%d] = %d, ", i, recv_data.counts[i]);
        for (int i = 0; i < recv_data.num_msgs; i++)
        {
            printf("Displs[%d] = %ld, ", i, recv_data.indptr[i]);
            for (int j = recv_data.indptr[i]; j < recv_data.indptr[i] + recv_data.counts[i]; j++)
                printf("recvbuf[%d] = %d, ", j, std_recv_vals[j]);
        }
        printf("\n");
    }

    // 2. Node-Aware Communication
    MPIX_Dist_graph_create_adjacent(MPI_COMM_WORLD,
            recv_data.num_msgs, 
            recv_data.procs.data(), 
            recv_data.counts.data(),
            send_data.num_msgs, 
            send_data.procs.data(),
            send_data.counts.data(),
            MPI_INFO_NULL, 
            0, 
            &neighbor_comm);
    MPIX_Neighbor_alltoallw_init(alltoallv_send_vals.data(), 
            send_data.counts.data(),
            send_data.indptr.data(), 
            sendtypes.data(),
            new_recv_vals.data(), 
            recv_data.counts.data(),
            recv_data.indptr.data(), 
            recvtypes.data(),
            neighbor_comm, 
            MPI_INFO_NULL,
            &neighbor_request);

    MPIX_Start(neighbor_request);
    MPIX_Wait(neighbor_request, &status);

    // 3. Compare std_recv_vals and nap_recv_vals
    for (int i = 0; i < recv_data.size_msgs; i++)
    {
        ASSERT_EQ(std_recv_vals[i], new_recv_vals[i]);
    }

    MPIX_Request_free(neighbor_request);
    MPIX_Comm_free(neighbor_comm);
    MPI_Comm_free(&std_comm);

    setenv("PPN", "16", 1);
}

