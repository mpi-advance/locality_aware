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
#include "mpipcl/mpipcl.h"

void test_partitioned(const char* filename, int n_vec)
{
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
    int n_parts = 10;
    // Single exchange test
    // Precv/Psend inits
    std::vector<MPIX_Prequest> sreqs;
    std::vector<MPIX_Prequest> rreqs;
    if (A.send_comm.n_msgs)
        sreqs.resize(A.send_comm.n_msgs);
    if (A.recv_comm.n_msgs)
        rreqs.resize(A.recv_comm.n_msgs);

    int proc;
    int start, end;
    int tag = 2948;
    for (int i = 0; i < A.recv_comm.n_msgs; i++)
    {
        proc = A.recv_comm.procs[i];
        start = A.recv_comm.ptr[i] * n_vec;
        end = A.recv_comm.ptr[i+1] * n_vec;
        MPIX_Precv_init(&(partd_recv_vals[start]), n_parts, (int)(end - start), MPI_INT, proc, tag,
                MPI_COMM_WORLD, MPI_INFO_NULL, &(rreqs[i]));
    }

    for (int i = 0; i < A.send_comm.n_msgs; i++)
    {
        proc = A.send_comm.procs[i];
        start = A.send_comm.ptr[i];
        end = A.send_comm.ptr[i+1];
        MPIX_Psend_init(&(alltoallv_send_vals[start * n_vec]), n_parts, (int)(end - start) * n_vec, MPI_INT, proc, tag,
                MPI_COMM_WORLD, MPI_INFO_NULL, &(sreqs[i]));
    }

    if (A.send_comm.n_msgs)
        MPIX_Startall(A.send_comm.n_msgs, sreqs.data());
    if (A.recv_comm.n_msgs)
        MPIX_Startall(A.recv_comm.n_msgs, rreqs.data());

    for (int i = 0; i < A.recv_comm.n_msgs; i++) {
        for (int j = 0; j < n_parts; j++) {
            MPIX_Pready(j, &sreqs[i]);
        }
    }

    if (A.send_comm.n_msgs)
        MPIX_Waitall(A.send_comm.n_msgs, rreqs.data(), MPI_STATUSES_IGNORE);
    if (A.recv_comm.n_msgs)
    	MPIX_Waitall(A.recv_comm.n_msgs, sreqs.data(), MPI_STATUSES_IGNORE);

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

    // Test SpM-Multivector
    int n_vec_sizes = 4;
    int vec_sizes[n_vec_sizes] = {1, 2, 10, 100};
    for (int i = 0; i < n_mats; i++)
        for (int j = 0; j < n_vec_sizes; j++)
            test_partitioned((mat_dir + test_matrices[i]).c_str(), vec_sizes[j]);
}
