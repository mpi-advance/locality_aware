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

#include "neighbor_data.hpp"

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

    // Initial communication info (standard)
    int local_size = 10000; // Number of variables each rank stores
    MPIX_Data<int> send_data;
    MPIX_Data<int> recv_data;
    form_initial_communicator(local_size, &send_data, &recv_data);
    std::vector<long> global_send_idx(send_data.size_msgs);
    std::vector<long> global_recv_idx(recv_data.size_msgs);
    form_global_indices(local_size, send_data, recv_data, global_send_idx, global_recv_idx);

    // Test correctness of communication
    std::vector<int> std_recv_vals(recv_data.size_msgs);
    std::vector<int> persistent_recv_vals(recv_data.size_msgs);
    std::vector<int> part_recv_vals(recv_data.size_msgs);
    std::vector<int> loc_recv_vals(recv_data.size_msgs);

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

    MPIX_Info* xinfo;
    MPIX_Info_init(&xinfo);

    // Standard MPI Dist Graph Create
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

    // MPI Advance Dist Graph Create
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

    // Update Locality : 4 PPN (for single-node tests)
    update_locality(neighbor_comm, 4);


    // Standard MPI Implementation of Alltoallv
    int* send_counts = send_data.counts.data();
    if (send_data.counts.data() == NULL)
        send_counts = new int[1];
    int* recv_counts = recv_data.counts.data();
    if (recv_data.counts.data() == NULL)
        recv_counts = new int[1];
    MPI_Neighbor_alltoallv(alltoallv_send_vals.data(), 
            send_counts,
            send_data.indptr.data(), 
            MPI_INT,
            std_recv_vals.data(), 
            recv_counts,
            recv_data.indptr.data(), 
            MPI_INT,
            std_comm);
    if (send_data.counts.data() == NULL)
        delete[] send_counts;
    if (recv_data.counts.data() == NULL)
        delete[] recv_counts;


    // Simple Persistent MPI Advance Implementation
    MPIX_Neighbor_alltoallv_init(alltoallv_send_vals.data(), 
            send_data.counts.data(),
            send_data.indptr.data(), 
            MPI_INT,
            persistent_recv_vals.data(), 
            recv_data.counts.data(),
            recv_data.indptr.data(), 
            MPI_INT,
            neighbor_comm, 
            xinfo,
            &neighbor_request);

    // Reorder during first send/recv
    neighbor_request->reorder = 1;
    MPIX_Start(neighbor_request);
    MPIX_Wait(neighbor_request, &status);
    for (int i = 0; i < recv_data.size_msgs; i++)
    {
        ASSERT_EQ(std_recv_vals[i], persistent_recv_vals[i]);
    }

    // Standard send/recv with reordered recvs
    MPIX_Start(neighbor_request);
    MPIX_Wait(neighbor_request, &status);
    for (int i = 0; i < recv_data.size_msgs; i++)
    {
        ASSERT_EQ(std_recv_vals[i], persistent_recv_vals[i]);
    }
    MPIX_Request_free(&neighbor_request);

    MPIX_Info_free(&xinfo);
    MPIX_Comm_free(&neighbor_comm);
    MPI_Comm_free(&std_comm);

}

