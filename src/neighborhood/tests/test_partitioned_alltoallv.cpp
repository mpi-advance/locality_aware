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
#include <omp.h>

#include "tests/sparse_mat.hpp"
#include "tests/par_binary_IO.hpp"
#include "mpipcl.h"


void partitioned_communicate(
    int n_vec,
    int n_recvs,
    int n_sends,
    int size_sends,
    std::vector<int>& idxs,
    std::vector<int>& x,
    std::vector<MPIP_Request>& sreqs,
    std::vector<MPIP_Request>& rreqs,
    std::vector<int>& send_vals
) {
    // Partitioned communication
    //printf("rank %d starting...\n", rank);
    if (n_sends)
        MPIP_Startall(n_sends, sreqs.data());
    if (n_recvs)
        MPIP_Startall(n_recvs, rreqs.data());

    // Packing
    //printf("rank %d packing...\n", rank);
    if (size_sends)
    {
        int sizes = 0;
        #pragma omp parallel
        {
            for (int i = 0; i < n_sends; i++) {
                #pragma omp for nowait schedule(static)
                for (int j = 0; j < sreqs[i].size; j++) {
                    int idx = idxs[(sizes + j) / n_vec];
                    send_vals[sizes + j] = x[(idx * n_vec) + ((sizes + j) % n_vec)];
                }

                MPIP_Pready(omp_get_thread_num(), &sreqs[i]);

                #pragma omp barrier

                #pragma omp master
                {
                    sizes += sreqs[i].size;
                }

                #pragma omp barrier
            }
        }
    }

    //printf("rank %d waiting...\n", rank);
    if (n_sends)
        MPIP_Waitall(n_sends, sreqs.data(), MPI_STATUSES_IGNORE);
    if (n_recvs)
        MPIP_Waitall(n_recvs, rreqs.data(), MPI_STATUSES_IGNORE);

    // Serial packing code for reference
    // if (A.send_comm.size_msgs)
    // {
        // for (int i = 0; i < A.send_comm.size_msgs; i++)
        // {
        //     idx = A.send_comm.idx[i];
        //     for (int j = 0; j < n_vec; j++)
        //         alltoallv_send_vals[(i * n_vec) + j] = x[(idx * n_vec) + j];
        // }
    // }
}

void SpMV(
    int n_vec,
    Mat& on_proc,
    Mat& off_proc,
    std::vector<int>& x,
    std::vector<int>& off_proc_x,
    std::vector<int>& b
) {
    std::vector<int> vals;
    vals.resize(n_vec);
    // Local multiplication
    int start, end;
    int data, col_idx;
    //printf("rank %d local spmv...\n", rank);
    for (int i = 0; i < on_proc.n_rows; i++)
    {
        start = on_proc.rowptr[i];
        end = on_proc.rowptr[i+1];
        for (int vec = 0; vec < n_vec; vec++) {
            vals[vec] = 0;
        }
        for (int j = start; j < end; j  ++)
        {
            data = on_proc.data[j];
            col_idx = on_proc.col_idx[j];
            for (int vec = 0; vec < n_vec; vec++) {
                vals[vec] += data * x[col_idx * n_vec + vec];
            }
        }
        for (int vec = 0; vec < n_vec; vec++) {
            b[i * n_vec + vec] = vals[vec];
        }
    }

    //printf("rank %d off-proc spmv...\n", rank);
    // Add product of off-proc columns and non-local values of x
    for (int i = 0; i < off_proc.n_rows; i++) // Should be the same loop size as local
    {
        start = off_proc.rowptr[i];
        end = off_proc.rowptr[i+1];
        for (int vec = 0; vec < n_vec; vec++) {
            vals[vec] = 0;
        }
        for (int j = start; j < end; j++)
        {
            data = off_proc.data[j];
            col_idx = off_proc.col_idx[j];
            for (int vec = 0; vec < n_vec; vec++) {
                vals[vec] += data * off_proc_x[col_idx * n_vec + vec];
            }
        }
        for (int vec = 0; vec < n_vec; vec++) {
            b[i * n_vec + vec] += vals[vec];
            x[i * n_vec + vec] = b[i * n_vec + vec];
        }
    }
}

void test_partitioned(const char* filename, int n_vec)
{
    // msg counts must be divisible by n_parts. simplest method is to have n_parts divide n_vec
    int n_parts = omp_get_max_threads();// omp_get_num_threads(); 

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    //printf("rank %d threads %d\n", rank, n_parts);

    // Read suitesparse matrix
    ParMat<int> A;
    readParMatrix(filename, A);
    form_comm(A);
    //printf("rank %d read matrix\n", rank);
    if (rank == 0 && A.global_rows != A.global_cols)
        printf("NOT SQUARE~~~~~~~~~~~~~~~~~~~~\n");

    std::vector<int> std_recv_vals, partd_recv_vals;
    std::vector<int> send_vals, alltoallv_send_vals;
    std::vector<int> x, b;

    if (A.on_proc.n_cols)
    {
        b.resize(A.on_proc.n_rows * n_vec);
        x.resize(A.on_proc.n_cols * n_vec);
        send_vals.resize(A.on_proc.n_cols * n_vec);
        std::iota(send_vals.begin(), send_vals.end(), 0);
        for (int i = 0; i < A.on_proc.n_cols * n_vec; i++)
        {
            send_vals[i] += (rank*1000);
            x[i] = send_vals[i];
        }
    }

    if (A.recv_comm.size_msgs)
    {
        std_recv_vals.resize(A.recv_comm.size_msgs * n_vec);
        partd_recv_vals.resize(A.recv_comm.size_msgs * n_vec);
    }

    if (A.send_comm.size_msgs)
    {
        alltoallv_send_vals.resize(A.send_comm.size_msgs * n_vec);
    }

    // Precv/Psend inits
    // Launch threads?
    // set x to be send_vals
    // Iterations: use x instead of send_vals
        // Communicate: For each message:
            // Start
            // Pack
            // Pready's
            // Wait/parrives?
            // unpack
        // Compute b
        // set x to b

    // Initialization
    //printf("rank %d initializing...\n", rank);
    std::vector<MPIP_Request> sreqs;
    std::vector<MPIP_Request> rreqs;
    if (A.send_comm.n_msgs)
        sreqs.resize(A.send_comm.n_msgs);
    if (A.recv_comm.n_msgs)
        rreqs.resize(A.recv_comm.n_msgs);

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
    //printf("rank %d initialized\n", rank);

    // Iterations
    int iters = 5;
    for (int iter = 0; iter < iters; iter++) {

        // Point-to-point communication
        communicate(A, x, std_recv_vals, MPI_INT, n_vec);

        partitioned_communicate(n_vec, A.recv_comm.n_msgs, A.send_comm.n_msgs, 
            A.send_comm.size_msgs, A.send_comm.idx, x, sreqs, rreqs, alltoallv_send_vals);

        //printf("rank %d verifying...\n", rank);
        for (int i = 0; i < A.recv_comm.size_msgs; i++)
        {
            ASSERT_EQ(std_recv_vals[i], partd_recv_vals[i]);
        }
        
        SpMV(n_vec, A.on_proc, A.off_proc, x, partd_recv_vals, b);
    }

    // Cleanup
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
        //"gams10a.pm",
        //"gams10am.pm",
        //"D_10.pm",
        "oscil_dcop_11.pm",
        "tumorAntiAngiogenesis_4.pm",
        //"ch5-5-b1.pm",
        "msc01050.pm",
        "SmaGri.pm",
        "radfr1.pm",
        //"bibd_49_3.pm",
        "can_1054.pm",
        "can_1072.pm",
        //"lp_sctap2.pm",
        //"lp_woodw.pm",
    };

    // Test SpM-Multivector
    std::vector<int> vec_sizes = {16, 128};
    for (size_t i = 0; i < test_matrices.size(); i++) {
        if (rank == 0) 
            printf("Matrix %d...\n", i);
        for (size_t j = 0; j < vec_sizes.size(); j++) {
            test_partitioned((mat_dir + test_matrices[i]).c_str(), vec_sizes[j]);
        }
    }
}
