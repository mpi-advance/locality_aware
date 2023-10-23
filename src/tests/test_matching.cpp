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

    // Read suitesparse matrix
    ParMat<int> A;
    int idx;
    readParMatrix(filename, A);

    // Form Communication Pattern
    // e.g. who do I send to, what do I send to them
    //      who do I recv from, what do I recv from them
    form_comm(A);

    // Timing Variables
    double t0, tfinal;
    int n_iter = 1000;  // May need to decrease for tests with large numbers of nonzeros per process 

    std::vector<int> recv_vals;
    std::vector<int> send_vals;

    // Creating send values (vals[i] = rank*1000 + i)
    if (A.on_proc.n_cols)
    {
        send_vals.resize(A.on_proc.n_cols);
        std::iota(send_vals.begin(), send_vals.end(), 0);
        for (int i = 0; i < A.on_proc.n_cols; i++)
            send_vals[i] += (rank*1000);
    }

    // Allocating recv_vals buffer
    if (A.recv_comm.size_msgs)
    {
        recv_vals.resize(A.recv_comm.size_msgs);
    }

    // TODO : Time order_comm
    t0 = MPI_Wtime();
    order_comm(A, send_vals, recv_vals, MPI_INT); 
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("order_comm time: %e\n", t0);

    std::vector<int> new_recv_vals;
    if (A.recv_comm.size_msgs)
    {
        new_recv_vals.resize(A.recv_comm.size_msgs);
    }

    // Time original communciate (warm-up first)
    communicate(A, send_vals, recv_vals, MPI_INT);

    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
        communicate(A, send_vals, recv_vals, MPI_INT);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Communication Time : %e\n", t0);

    // Time new communicate (warm-up first)
    communicate2(A, send_vals, new_recv_vals, MPI_INT);

    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
        communicate2(A, send_vals, new_recv_vals, MPI_INT);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Communication 2 Time: %e\n", t0);


    // Check for correctness
    communicate(A, send_vals, recv_vals, MPI_INT);
    communicate2(A, send_vals, new_recv_vals, MPI_INT);
    for (int i = 0; i < A.recv_comm.size_msgs; i++)
        ASSERT_EQ(recv_vals[i], new_recv_vals[i]);

    // TODO : 
    //    1. Add a new method similar to communicate, but instead of posting MPI_Irecv for each message
    //        go through n_recvs msgs, call MPI_Probe for the message, dynamically find the sending process and msg_size
    //        and recv based on that information (tracking which message arrived at each probe)
    //    2. Reorder the recv_comm by the order in which the messages arrived in step 1
    //    3. Time another communicate loop (will now be using this reordered recv_comm)
    
    //    Hint : to see an example of probe/dynamic receiving, check out the 
    //        form_send_comm methods in src/test/sparse_mat.hpp
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

    //test_matrix("../../../test_data/dwt_162.pm");
    //test_matrix("../../../test_data/odepa400.pm");
    //test_matrix("../../../test_data/ww_36_pmec_36.pm");
    //test_matrix("../../../test_data/bcsstk01.pm");
    //test_matrix("../../../test_data/west0132.pm");
    //test_matrix("../../../test_data/gams10a.pm");
    //test_matrix("../../../test_data/gams10am.pm");
    //test_matrix("../../../test_data/D_10.pm");
    //test_matrix("../../../test_data/oscil_dcop_11.pm");
    //test_matrix("../../../test_data/tumorAntiAngiogenesis_4.pm");
    //test_matrix("../../../test_data/ch5-5-b1.pm");
    //test_matrix("../../../test_data/msc01050.pm");
    //test_matrix("../../../test_data/SmaGri.pm");
    //test_matrix("../../../test_data/radfr1.pm");
    //test_matrix("../../../test_data/bibd_49_3.pm");
    //test_matrix("../../../test_data/can_1054.pm");
    //test_matrix("../../../test_data/can_1072.pm");
    //test_matrix("../../../test_data/lp_sctap2.pm");
    test_matrix("../../../test_data/lp_woodw.pm");
}

