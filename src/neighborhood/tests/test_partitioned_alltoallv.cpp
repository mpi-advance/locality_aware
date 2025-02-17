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
#include "mpipcl.h"

void test_partitioned(const char* filename, int n_vec)
{
    // msg counts must be divisible by n_parts. simplest method is to have n_parts divide n_vec
    int n_parts = 2; 

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Read suitesparse matrix
    ParMat<int> A;
    int idx;
    readParMatrix(filename, A);
    form_comm(A);

    std::vector<int> std_recv_vals, partd_recv_vals;
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
        partd_recv_vals.resize(A.recv_comm.size_msgs * n_vec);
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

    // Point-to-point communication
    communicate(A, send_vals, std_recv_vals, MPI_INT, n_vec);

    // Partitioned communication
    // Single exchange test
    // Precv/Psend inits
    std::vector<MPIP_Request> sreqs;
    std::vector<MPIP_Request> rreqs;
    if (A.send_comm.n_msgs)
        sreqs.resize(A.send_comm.n_msgs);
    if (A.recv_comm.n_msgs)
        rreqs.resize(A.recv_comm.n_msgs);

    //printf("rank %d initializing...\n", rank);
    int proc;
    int start, end;
    int tag = 2949;
    for (int i = 0; i < A.recv_comm.n_msgs; i++)
    {
        proc = A.recv_comm.procs[i];
        start = A.recv_comm.ptr[i] * n_vec;
        end = A.recv_comm.ptr[i+1] * n_vec;
        MPIP_Precv_init(&(partd_recv_vals[start]), n_parts, (int)(end - start)/n_parts, MPI_INT, proc, tag,
                MPI_COMM_WORLD, MPI_INFO_NULL, &(rreqs[i]));
    }

    for (int i = 0; i < A.send_comm.n_msgs; i++)
    {
        proc = A.send_comm.procs[i];
        start = A.send_comm.ptr[i] * n_vec;
        end = A.send_comm.ptr[i+1] * n_vec;
        MPIP_Psend_init(&(alltoallv_send_vals[start]), n_parts, (int)(end - start)/n_parts, MPI_INT, proc, tag,
                MPI_COMM_WORLD, MPI_INFO_NULL, &(sreqs[i]));
    }

    //printf("rank %d starting...\n", rank);
    if (A.send_comm.n_msgs)
        MPIP_Startall(A.send_comm.n_msgs, sreqs.data());
    if (A.recv_comm.n_msgs)
        MPIP_Startall(A.recv_comm.n_msgs, rreqs.data());

    //printf("rank %d marking ready...\n", rank);
    for (int i = 0; i < A.send_comm.n_msgs; i++) {
        for (int j = 0; j < n_parts; j++) {
            MPIP_Pready(j, &sreqs[i]);
        }
    }

    //printf("rank %d waiting...\n", rank);
    if (A.send_comm.n_msgs)
        MPIP_Waitall(A.send_comm.n_msgs, sreqs.data(), MPI_STATUSES_IGNORE);
    if (A.recv_comm.n_msgs)
        MPIP_Waitall(A.recv_comm.n_msgs, rreqs.data(), MPI_STATUSES_IGNORE);

    //printf("rank %d verifying...\n", rank);
    for (int i = 0; i < A.recv_comm.size_msgs; i++)
    {
        ASSERT_EQ(std_recv_vals[i], partd_recv_vals[i]);
    }
    // Precv/Psend inits
    // Launch threads?
    // Iterations:
        // Communicate: For each message:
            // Start
            // Pack
            // Pready's
            // Wait/parrives?
            // unpack
        // Compute

    //printf("rank %d freeing...\n", rank);
    for (int i = 0; i < A.recv_comm.n_msgs; i++) {
        MPIP_Request_free(&rreqs[i]);
    }
    for (int i = 0; i < A.send_comm.n_msgs; i++) {
        MPIP_Request_free(&sreqs[i]);
    }
    //printf("rank %d complete\n", rank);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp=RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
}


TEST(RandomCommTest, TestsInTests)
{
    // Get MPI Information
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::string mat_dir = "../../../../test_data/";
    std::vector<std::string> test_matrices = {
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

    // Test SpM-Multivector
    std::vector<int> vec_sizes = {2, 16, 128};
    for (size_t i = 0; i < test_matrices.size(); i++) {
        // if (rank == 0) 
        //     printf("Matrix %d...\n", i);
        for (size_t j = 0; j < vec_sizes.size(); j++) {
            test_partitioned((mat_dir + test_matrices[i]).c_str(), vec_sizes[j]);
        }
    }
}
