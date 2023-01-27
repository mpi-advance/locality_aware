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

#include "sparse_mat.hpp"
#include "par_binary_IO.hpp"

void test_matrix(const char* filename)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Read suitesparse matrix
    ParMat<int> A;
    int idx;
    readParMatrix(filename, A);
    form_comm(A);

    std::vector<int> send_vals;
    if (A.on_proc.n_rows)
    {
        send_vals.resize(A.on_proc.n_rows);
        std::iota(send_vals.begin(), send_vals.end(), 0);
        for (int i = 0; i < A.on_proc.n_rows; i++)
            send_vals[i] += (rank*1000);
    }
    std::vector<int> alltoallv_send_vals;
    if (A.send_comm.size_msgs)
    {
        alltoallv_send_vals.resize(A.send_comm.size_msgs);
        for (int i = 0; i < A.send_comm.size_msgs; i++)
        {
            idx = A.send_comm.idx[i];
if (idx >= A.on_proc.n_rows) printf("Idx[%d] %d, NRows %d\n", i, idx, A.on_proc.n_rows);
            alltoallv_send_vals[i] = send_vals[idx];
        }
    }

/*
    std::vector<int> std_recv_vals, neigh_recv_vals, new_recv_vals,
            locality_recv_vals, part_locality_recv_vals;
    if (A.recv_comm.size_msgs)
    {
        std_recv_vals.resize(A.recv_comm.size_msgs);
        neigh_recv_vals.resize(A.recv_comm.size_msgs);
        new_recv_vals.resize(A.recv_comm.size_msgs);
        locality_recv_vals.resize(A.recv_comm.size_msgs);
        part_locality_recv_vals.resize(A.recv_comm.size_msgs);
    }

    communicate(A, send_vals, std_recv_vals, MPI_INT);

    MPI_Comm std_comm;
    MPI_Status status;
    MPIX_Comm* neighbor_comm;
    MPIX_Request* neighbor_request;

    MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD,
            A.recv_comm.n_msgs,
            A.recv_comm.procs.data(), 
            A.recv_comm.counts.data(),
            A.send_comm.n_msgs, 
            A.send_comm.procs.data(),
            A.send_comm.counts.data(),
            MPI_INFO_NULL, 
            0, 
            &std_comm);
    MPI_Neighbor_alltoallv(alltoallv_send_vals.data(), 
            A.send_comm.counts.data(),
            A.send_comm.ptr.data(), 
            MPI_INT,
            neigh_recv_vals.data(), 
            A.recv_comm.counts.data(),
            A.recv_comm.ptr.data(), 
            MPI_INT,
            std_comm);

    // 3. Compare std_recv_vals and nap_recv_vals
    for (int i = 0; i < A.recv_comm.size_msgs; i++)
    {
        ASSERT_EQ(std_recv_vals[i], neigh_recv_vals[i]);
    }

    // 2. Node-Aware Communication
    MPIX_Dist_graph_create_adjacent(MPI_COMM_WORLD,
            A.recv_comm.n_msgs,
            A.recv_comm.procs.data(), 
            A.recv_comm.counts.data(),
            A.send_comm.n_msgs, 
            A.send_comm.procs.data(),
            A.send_comm.counts.data(),
            MPI_INFO_NULL, 
            0, 
            &neighbor_comm);
    update_locality(neighbor_comm, 4);
    
    MPIX_Neighbor_alltoallv_init(alltoallv_send_vals.data(), 
            A.send_comm.counts.data(),
            A.send_comm.ptr.data(), 
            MPI_INT,
            new_recv_vals.data(), 
            A.recv_comm.counts.data(),
            A.recv_comm.ptr.data(), 
            MPI_INT,
            neighbor_comm, 
            MPI_INFO_NULL,
            &neighbor_request);

    MPIX_Start(neighbor_request);
    MPIX_Wait(neighbor_request, &status);
    MPIX_Request_free(neighbor_request);

    // 3. Compare std_recv_vals and nap_recv_vals
    for (int i = 0; i < A.recv_comm.size_msgs; i++)
    {
        ASSERT_EQ(std_recv_vals[i], new_recv_vals[i]);
    }


    // 3. MPI Advance - Optimized Communication
    std::vector<long> send_indices(A.send_comm.size_msgs);
    for (int i = 0; i < A.send_comm.size_msgs; i++)
        send_indices[i] = A.send_comm.idx[i] + A.first_row;

    MPIX_Neighbor_locality_alltoallv_init(alltoallv_send_vals.data(), 
            A.send_comm.counts.data(),
            A.send_comm.ptr.data(), 
            send_indices.data(),
            MPI_INT,
            locality_recv_vals.data(), 
            A.recv_comm.counts.data(),
            A.recv_comm.ptr.data(), 
            A.off_proc_columns.data(),
            MPI_INT,
            neighbor_comm, 
            MPI_INFO_NULL,
            &neighbor_request);


    MPIX_Start(neighbor_request);
    MPIX_Wait(neighbor_request, &status);
    MPIX_Request_free(neighbor_request);
    // 3. Compare std_recv_vals and nap_recv_vals
    for (int i = 0; i < A.recv_comm.size_msgs; i++)
    {
        ASSERT_EQ(std_recv_vals[i], locality_recv_vals[i]);
    }

    MPIX_Neighbor_part_locality_alltoallv_init(alltoallv_send_vals.data(), 
            A.send_comm.counts.data(),
            A.send_comm.ptr.data(), 
            MPI_INT,
            part_locality_recv_vals.data(), 
            A.recv_comm.counts.data(),
            A.recv_comm.ptr.data(), 
            MPI_INT,
            neighbor_comm, 
            MPI_INFO_NULL,
            &neighbor_request);

    MPIX_Start(neighbor_request);
    MPIX_Wait(neighbor_request, &status);
    MPIX_Request_free(neighbor_request);

    for (int i = 0; i < A.recv_comm.size_msgs; i++)
    {
        ASSERT_EQ(std_recv_vals[i], part_locality_recv_vals[i]);
    }

    MPIX_Comm_free(neighbor_comm);
    MPI_Comm_free(&std_comm);
*/
}

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

//    test_matrix("../../../../test_data/dwt_162.pm");
//    test_matrix("../../../../test_data/odepa400.pm");
//    test_matrix("../../../../test_data/ww_36_pmec_36.pm");
//    test_matrix("../../../../test_data/bcsstk01.pm");
//    test_matrix("../../../../test_data/west0132.pm");
    test_matrix("../../../../test_data/gams10a.pm");
//    test_matrix("../../../../test_data/gams10am.pm");
//    test_matrix("../../../../test_data/ch5-5-b1.pm");
}

