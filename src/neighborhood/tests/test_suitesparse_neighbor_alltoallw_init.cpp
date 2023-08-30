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
    MPI_Comm_rank(MPI_COMM_WORLD, &num_procs);

    // Read suitesparse matrix
    ParMat<MPI_Aint> A;
    readParMatrix(filename, A);
    allocate_comm(A);
    form_comm(A);

    std::vector<int> send_vals(A.on_proc.n_rows);
    std::iota(send_vals.begin(), send_vals.end(), 0);
    std::vector<int> alltoallv_send_vals(A.send_comm->size_msgs);
    for (int i = 0; i < A.send_comm->size_msgs; i++)
        alltoallv_send_vals[i] = send_vals[A.send_comm->idx[i]];

    std::vector<int> std_recv_vals(A.recv_comm->size_msgs);
    std::vector<int> neigh_recv_vals(A.recv_comm->size_msgs);
    std::vector<int> new_recv_vals(A.recv_comm->size_msgs);

    communicate(A, send_vals, std_recv_vals, MPI_INT);


    int int_size = sizeof(int);
    for (int i = 0; i < A.send_comm->n_msgs; i++)
        A.send_comm->ptr[i+1] *= int_size;
    for (int i = 0; i < A.recv_comm->n_msgs; i++)
        A.recv_comm->ptr[i+1] *= int_size;


    std::vector<MPI_Datatype> sendtypes(A.send_comm->n_msgs, MPI_INT);
    std::vector<MPI_Datatype> recvtypes(A.recv_comm->n_msgs, MPI_INT);

    MPI_Status status;
    MPI_Comm std_comm;

    MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD,
            A.recv_comm->n_msgs,
            A.recv_comm->procs.data(), 
            A.recv_comm->counts.data(),
            A.send_comm->n_msgs, 
            A.send_comm->procs.data(),
            A.send_comm->counts.data(),
            MPI_INFO_NULL, 
            0, 
            &std_comm);
    MPI_Neighbor_alltoallw(alltoallv_send_vals.data(), 
            A.send_comm->counts.data(),
            A.send_comm->ptr.data(), 
            sendtypes.data(),
            neigh_recv_vals.data(), 
            A.recv_comm->counts.data(),
            A.recv_comm->ptr.data(), 
            recvtypes.data(),
            std_comm);

    // 3. Compare std_recv_vals and nap_recv_vals
    for (int i = 0; i < A.recv_comm->size_msgs; i++)
    {
        ASSERT_EQ(std_recv_vals[i], neigh_recv_vals[i]);
    }
    MPI_Comm_free(&std_comm);
    

    // 2. Node-Aware Communication
    MPIX_Comm* neighbor_comm;
    MPIX_Request* neighbor_request;
    MPIX_Dist_graph_create_adjacent(MPI_COMM_WORLD,
            A.recv_comm->n_msgs,
            A.recv_comm->procs.data(), 
            A.recv_comm->counts.data(),
            A.send_comm->n_msgs, 
            A.send_comm->procs.data(),
            A.send_comm->counts.data(),
            MPI_INFO_NULL, 
            0, 
            &neighbor_comm);
    update_locality(neighbor_comm, 4);
    MPIX_Neighbor_alltoallw_init(alltoallv_send_vals.data(), 
            A.send_comm->counts.data(),
            A.send_comm->ptr.data(), 
            sendtypes.data(),
            new_recv_vals.data(), 
            A.recv_comm->counts.data(),
            A.recv_comm->ptr.data(), 
            recvtypes.data(),
            neighbor_comm, 
            MPI_INFO_NULL,
            &neighbor_request);

    MPIX_Start(neighbor_request);
    MPIX_Wait(neighbor_request, &status);

    // 3. Compare std_recv_vals and nap_recv_vals
    for (int i = 0; i < A.recv_comm->size_msgs; i++)
    {
        ASSERT_EQ(std_recv_vals[i], new_recv_vals[i]);
    }
    free_comm(A);
    MPIX_Request_free(neighbor_request);
    MPIX_Comm_free(neighbor_comm);

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

    setenv("PPN", "4", 1);

    test_matrix("../../../../test_data/dwt_162.pm");
    test_matrix("../../../../test_data/odepa400.pm");
    test_matrix("../../../../test_data/ww_36_pmec_36.pm");

    setenv("PPN", "16", 1);
}

