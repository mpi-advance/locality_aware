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
#include <string>

#include "tests/sparse_mat.hpp"
#include "tests/par_binary_IO.hpp"

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

    std::vector<int> std_recv_vals, neigh_recv_vals, new_recv_vals,
            locality_recv_vals, part_locality_recv_vals;
    std::vector<int> send_vals, alltoallv_send_vals;
    std::vector<long> send_indices;

    if (A.on_proc.n_cols)
    {
        send_vals.resize(A.on_proc.n_cols);
        std::iota(send_vals.begin(), send_vals.end(), 0);
        for (int i = 0; i < A.on_proc.n_cols; i++)
            send_vals[i] += (rank*1000);
    }

    if (A.recv_comm.size_msgs)
    {
        std_recv_vals.resize(A.recv_comm.size_msgs);
        neigh_recv_vals.resize(A.recv_comm.size_msgs);
        new_recv_vals.resize(A.recv_comm.size_msgs);
        locality_recv_vals.resize(A.recv_comm.size_msgs);
        part_locality_recv_vals.resize(A.recv_comm.size_msgs);
    }

    if (A.send_comm.size_msgs)
    {
        alltoallv_send_vals.resize(A.send_comm.size_msgs);
        send_indices.resize(A.send_comm.size_msgs);
        for (int i = 0; i < A.send_comm.size_msgs; i++)
        {
            idx = A.send_comm.idx[i];
            alltoallv_send_vals[i] = send_vals[idx];
            send_indices[i] = A.send_comm.idx[i] + A.first_col;
        }
    }

    communicate(A, send_vals, std_recv_vals, MPI_INT);

    MPI_Comm std_comm;
    MPI_Status status;
    MPIX_Comm* neighbor_comm;
    MPIX_Request* neighbor_request;
    MPIX_Info* xinfo;

    MPIX_Info_init(&xinfo);

    int* s = A.recv_comm.procs.data();
    if (A.recv_comm.n_msgs == 0)
        s = MPI_WEIGHTS_EMPTY;
    int* d = A.send_comm.procs.data();
    if (A.send_comm.n_msgs  == 0)
        d = MPI_WEIGHTS_EMPTY;

    PMPI_Dist_graph_create_adjacent(MPI_COMM_WORLD,
            A.recv_comm.n_msgs,
            s,
            MPI_UNWEIGHTED,
            A.send_comm.n_msgs,
            d,
            MPI_UNWEIGHTED,
            MPI_INFO_NULL,
            0,
            &std_comm);

    int* send_counts = A.send_comm.counts.data();
    if (A.send_comm.counts.data() == NULL)
        send_counts = new int[1];
    int* recv_counts = A.recv_comm.counts.data();
    if (A.recv_comm.counts.data() == NULL)
        recv_counts = new int[1];
    PMPI_Neighbor_alltoallv(alltoallv_send_vals.data(),
            send_counts,
            A.send_comm.ptr.data(),
            MPI_INT,
            neigh_recv_vals.data(),
            recv_counts,
            A.recv_comm.ptr.data(),
            MPI_INT,
            std_comm);
    if (A.send_comm.counts.data() == NULL)
        delete[] send_counts;
    if (A.recv_comm.counts.data() == NULL)
        delete[] recv_counts;

    MPIX_Dist_graph_create_adjacent(MPI_COMM_WORLD,
            A.recv_comm.n_msgs,
            A.recv_comm.procs.data(),
            MPI_UNWEIGHTED,
            A.send_comm.n_msgs,
            A.send_comm.procs.data(),
            MPI_UNWEIGHTED,
            MPI_INFO_NULL,
            0,
            &neighbor_comm);

    update_locality(neighbor_comm, 4);


    MPIX_Neighbor_alltoallv(alltoallv_send_vals.data(),
            A.send_comm.counts.data(),
            A.send_comm.ptr.data(),
            MPI_INT,
            neigh_recv_vals.data(),
            A.recv_comm.counts.data(),
            A.recv_comm.ptr.data(),
            MPI_INT,
            neighbor_comm);

    for (int i = 0; i < A.recv_comm.size_msgs; i++)
    {
        ASSERT_EQ(std_recv_vals[i], neigh_recv_vals[i]);
    }

    // 2. Node-Aware Communication
    MPIX_Neighbor_alltoallv_init(alltoallv_send_vals.data(),
            A.send_comm.counts.data(),
            A.send_comm.ptr.data(),
            MPI_INT,
            new_recv_vals.data(),
            A.recv_comm.counts.data(),
            A.recv_comm.ptr.data(),
            MPI_INT,
            neighbor_comm,
            xinfo,
            &neighbor_request);

    MPIX_Start(neighbor_request);
    MPIX_Wait(neighbor_request, &status);
    MPIX_Request_free(&neighbor_request);

    // 3. Compare std_recv_vals and nap_recv_vals
    for (int i = 0; i < A.recv_comm.size_msgs; i++)
    {
        ASSERT_EQ(std_recv_vals[i], new_recv_vals[i]);
    }

    // 3. MPI Advance - Optimized Communication
    MPIX_Neighbor_part_locality_alltoallv_init(alltoallv_send_vals.data(),
            A.send_comm.counts.data(),
            A.send_comm.ptr.data(),
            MPI_INT,
            part_locality_recv_vals.data(),
            A.recv_comm.counts.data(),
            A.recv_comm.ptr.data(),
            MPI_INT,
            neighbor_comm,
            xinfo,
            &neighbor_request);

    MPIX_Start(neighbor_request);
    MPIX_Wait(neighbor_request, &status);
    MPIX_Request_free(&neighbor_request);

    for (int i = 0; i < A.recv_comm.size_msgs; i++)
    {
        ASSERT_EQ(std_recv_vals[i], part_locality_recv_vals[i]);
    }

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
            xinfo,
            &neighbor_request);

    MPIX_Start(neighbor_request);
    MPIX_Wait(neighbor_request, &status);
    MPIX_Request_free(&neighbor_request);

    // 3. Compare std_recv_vals and nap_recv_vals
    for (int i = 0; i < A.recv_comm.size_msgs; i++)
    {
        ASSERT_EQ(std_recv_vals[i], locality_recv_vals[i]);
    }

    MPIX_Info_free(&xinfo);
    MPIX_Comm_free(&neighbor_comm);
    PMPI_Comm_free(&std_comm);
}

void test_multivector(const char* filename, int n_vec)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Read suitesparse matrix
    ParMat<int> A;
    int idx;
    readParMatrix(filename, A);
    form_comm(A);

    std::vector<int> std_recv_vals, neigh_recv_vals, new_recv_vals;
    std::vector<int> send_vals, alltoallv_send_vals;

    if (A.on_proc.n_cols)
    {
        send_vals.resize(A.on_proc.n_cols * n_vec);
        std::iota(send_vals.begin(), send_vals.end(), 0);
        for (int i = 0; i < A.on_proc.n_cols * n_vec; i++)
            send_vals[i] += (rank*1000);
    }

    if (A.recv_comm.size_msgs)
    {
        std_recv_vals.resize(A.recv_comm.size_msgs * n_vec);
        neigh_recv_vals.resize(A.recv_comm.size_msgs * n_vec);
        new_recv_vals.resize(A.recv_comm.size_msgs * n_vec);
    }

    if (A.send_comm.size_msgs)
    {
        alltoallv_send_vals.resize(A.send_comm.size_msgs * n_vec);
        for (int i = 0; i < A.send_comm.size_msgs; i++)
        {
            idx = A.send_comm.idx[i];
            for (int j = 0; j < n_vec; j++)
                alltoallv_send_vals[(i * n_vec) + j] = send_vals[(idx * n_vec) + j];
        }
    }

    communicate(A, send_vals, std_recv_vals, MPI_INT, n_vec);

    MPI_Comm std_comm;
    MPI_Status status;
    MPIX_Comm* neighbor_comm;
    MPIX_Request* neighbor_request;
    MPIX_Info* xinfo;

    MPIX_Info_init(&xinfo);

    int* s = A.recv_comm.procs.data();
    if (A.recv_comm.n_msgs == 0)
        s = MPI_WEIGHTS_EMPTY;
    int* d = A.send_comm.procs.data();
    if (A.send_comm.n_msgs  == 0)
        d = MPI_WEIGHTS_EMPTY;

    PMPI_Dist_graph_create_adjacent(MPI_COMM_WORLD,
            A.recv_comm.n_msgs,
            s,
            MPI_UNWEIGHTED,
            A.send_comm.n_msgs,
            d,
            MPI_UNWEIGHTED,
            MPI_INFO_NULL,
            0,
            &std_comm);

    int* send_counts = A.send_comm.counts.data();
    if (A.send_comm.counts.data() == NULL)
        send_counts = new int[1];
    int* recv_counts = A.recv_comm.counts.data();
    if (A.recv_comm.counts.data() == NULL)
        recv_counts = new int[1];

    // multiplied n_vec values are used for rest of test
    for (int i = 0; i < A.send_comm.n_msgs; i++) {
        A.send_comm.counts[i] *= n_vec;
        A.send_comm.ptr[i] *= n_vec;
    }
    for (int i = 0; i < A.recv_comm.n_msgs; i++) {
        A.recv_comm.counts[i] *= n_vec;
        A.recv_comm.ptr[i] *= n_vec;
    }

    PMPI_Neighbor_alltoallv(alltoallv_send_vals.data(),
            send_counts,
            A.send_comm.ptr.data(),
            MPI_INT,
            neigh_recv_vals.data(),
            recv_counts,
            A.recv_comm.ptr.data(),
            MPI_INT,
            std_comm);
    if (A.send_comm.counts.data() == NULL)
        delete[] send_counts;
    if (A.recv_comm.counts.data() == NULL)
        delete[] recv_counts;


    for (int i = 0; i < A.recv_comm.size_msgs * n_vec; i++)
    {
        ASSERT_EQ(std_recv_vals[i], neigh_recv_vals[i]);
    }

    MPIX_Dist_graph_create_adjacent(MPI_COMM_WORLD,
            A.recv_comm.n_msgs,
            A.recv_comm.procs.data(),
            MPI_UNWEIGHTED,
            A.send_comm.n_msgs,
            A.send_comm.procs.data(),
            MPI_UNWEIGHTED,
            MPI_INFO_NULL,
            0,
            &neighbor_comm);

    update_locality(neighbor_comm, 4);

    MPIX_Neighbor_alltoallv(alltoallv_send_vals.data(),
            A.send_comm.counts.data(),
            A.send_comm.ptr.data(),
            MPI_INT,
            neigh_recv_vals.data(),
            A.recv_comm.counts.data(),
            A.recv_comm.ptr.data(),
            MPI_INT,
            neighbor_comm);

    for (int i = 0; i < A.recv_comm.size_msgs; i++)
    {
        ASSERT_EQ(std_recv_vals[i], neigh_recv_vals[i]);
    }

    // MPIX Communication
    MPIX_Neighbor_alltoallv_init(alltoallv_send_vals.data(),
            A.send_comm.counts.data(),
            A.send_comm.ptr.data(),
            MPI_INT,
            new_recv_vals.data(),
            A.recv_comm.counts.data(),
            A.recv_comm.ptr.data(),
            MPI_INT,
            neighbor_comm,
            xinfo,
            &neighbor_request);

    MPIX_Start(neighbor_request);
    MPIX_Wait(neighbor_request, &status);

    for (int i = 0; i < A.recv_comm.size_msgs; i++)
    {
        ASSERT_EQ(std_recv_vals[i], new_recv_vals[i]);
    }

    // reset counts and pointers
    for (int i = 0; i < A.send_comm.n_msgs; i++) {
        A.send_comm.counts[i] /= n_vec;
        A.send_comm.ptr[i] /= n_vec;
    }
    for (int i = 0; i < A.recv_comm.n_msgs; i++) {
        A.recv_comm.counts[i] /= n_vec;
        A.recv_comm.ptr[i] /= n_vec;
    }

    MPIX_Request_free(&neighbor_request);
    MPIX_Info_free(&xinfo);
    MPIX_Comm_free(&neighbor_comm);
    PMPI_Comm_free(&std_comm);
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

    std::string mat_dir = "../../../../test_data/";
    int n_mats = 19;
    std::string test_matrices[n_mats] = {
        "dwt_162.pm",
        "odepa400.pm",
        "ww_36_pmec_36.pm",
        "bcsstk01.pm",
        "west0132.pm",
        "gams10a.pm",
        "gams10am.pm",
        "D_10.pm",
        "oscil_dcop_11.pm",
        "tumorAntiAngiogenesis_4.pm",
        "ch5-5-b1.pm",
        "msc01050.pm",
        "SmaGri.pm",
        "radfr1.pm",
        "bibd_49_3.pm",
        "can_1054.pm",
        "can_1072.pm",
        "lp_sctap2.pm",
        "lp_woodw.pm",
    };

    // Test SpMV
    for (int i = 0; i < n_mats; i++)
        test_matrix((mat_dir + test_matrices[i]).c_str());

    // Test SpM-Multivector
    int n_vec_sizes = 4;
    int vec_sizes[n_vec_sizes] = {1, 2, 10, 100};
    for (int i = 0; i < n_mats; i++)
        for (int j = 0; j < n_vec_sizes; j++)
            test_multivector((mat_dir + test_matrices[i]).c_str(), vec_sizes[j]);
}
