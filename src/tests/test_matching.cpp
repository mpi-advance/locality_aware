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
#include <dirent.h>
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
    // Order recv procs for communicate correctness test
    order_comm(A, send_vals, recv_vals, MPI_INT); 

    std::vector<int> new_recv_vals;
    if (A.recv_comm.size_msgs)
    {
        new_recv_vals.resize(A.recv_comm.size_msgs);
    } 

    // Time original communciate (warm-up first)
    communicate(A, send_vals, recv_vals, MPI_INT);
    
    MPI_Barrier(MPI_COMM_WORLD);   
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
        communicate(A, send_vals, recv_vals, MPI_INT);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%s Single Communicate Time: %e\n", filename, t0/n_iter);
/*
    MPI_Barrier(MPI_COMM_WORLD);   
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
        communicate(A, send_vals, recv_vals, MPI_INT);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%s Single Communicate Time (2nd): %e\n", filename, t0/n_iter);

    MPI_Barrier(MPI_COMM_WORLD);   
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
        communicate(A, send_vals, recv_vals, MPI_INT);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%s Single Communicate Time (3rd): %e\n", filename, t0/n_iter);
*/


    // Time new communicate (warm-up first)
    communicate2(A, send_vals, new_recv_vals, MPI_INT);

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
        communicate2(A, send_vals, new_recv_vals, MPI_INT);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%s Single Communicate 2 Time: %e\n", filename, t0/n_iter);
/*
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
        communicate2(A, send_vals, new_recv_vals, MPI_INT);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%s Single Communicate 2 Time (2nd): %e\n", filename, t0/n_iter);

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
        communicate2(A, send_vals, new_recv_vals, MPI_INT);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%s Single Communicate 2 Time (3rd): %e\n", filename, t0/n_iter);
*/


    // Check for correctness
    communicate(A, send_vals, recv_vals, MPI_INT);
    communicate2(A, send_vals, new_recv_vals, MPI_INT);
    for (int i = 0; i < A.recv_comm.size_msgs; i++)
        ASSERT_EQ(recv_vals[i], new_recv_vals[i]);

    // Order comm for timing
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        order_comm(A, send_vals, recv_vals, MPI_INT);
    }
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%s Order comm time (1 order_comm): %e\n", filename, t0/n_iter);

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
    test_matrix("../../../output-p/adder_dcop_02.pm");
    test_matrix("../../../output-p/barth5.pm");
    test_matrix("../../../output-p/bcsstk27.pm");
    test_matrix("../../../output-p/bcsstk34.pm");
    test_matrix("../../../output-p/bcsstm02.pm");
    test_matrix("../../../output-p/bfwb782.pm");
    test_matrix("../../../output-p/bp_800.pm");
    test_matrix("../../../output-p/bwm200.pm");
    test_matrix("../../../output-p/c-67.pm");
    test_matrix("../../../output-p/can_144.pm");
    test_matrix("../../../output-p/can_256.pm");
    test_matrix("../../../output-p/cavity24.pm");
    test_matrix("../../../output-p/cell1.pm");
    test_matrix("../../../output-p/Chem97ZtZ.pm");
    test_matrix("../../../output-p/dwt_607.pm");
    test_matrix("../../../output-p/ex32.pm");
    test_matrix("../../../output-p/ex36.pm");
    test_matrix("../../../output-p/fpga_dcop_14.pm");
    test_matrix("../../../output-p/fpga_dcop_18.pm");
    test_matrix("../../../output-p/fpga_dcop_46.pm");
    test_matrix("../../../output-p/fpga_trans_02.pm");
    test_matrix("../../../output-p/fs_680_3.pm");
    test_matrix("../../../output-p/fxm3_6.pm");
    test_matrix("../../../output-p/G26.pm");
    test_matrix("../../../output-p/G40.pm");
    test_matrix("../../../output-p/G49.pm");
    test_matrix("../../../output-p/G50.pm");
    test_matrix("../../../output-p/g7jac180.pm");
    test_matrix("../../../output-p/GD06_theory.pm");
    test_matrix("../../../output-p/hcircuit.pm");
    test_matrix("../../../output-p/lp_czprob.pm");
    test_matrix("../../../output-p/lpi_cplex2.pm");
    test_matrix("../../../output-p/lpi_klein1.pm");
    test_matrix("../../../output-p/lp_kb2.pm");
    test_matrix("../../../output-p/lp_nug06.pm");
     test_matrix("../../../output-p/lp_sc105.pm");
    test_matrix("../../../output-p/lp_standmps.pm");
    test_matrix("../../../output-p/lp_stocfor3.pm");
    test_matrix("../../../output-p/lp_tuff.pm");
    test_matrix("../../../output-p/lshp1009.pm");
     test_matrix("../../../output-p/oscil_dcop_20.pm");
    test_matrix("../../../output-p/oscil_trans_01.pm");
    test_matrix("../../../output-p/pde900.pm");
    test_matrix("../../../output-p/rajat03.pm");
    test_matrix("../../../output-p/rajat11.pm");
     test_matrix("../../../output-p/sinc12.pm");
    test_matrix("../../../output-p/str_600.pm");
    test_matrix("../../../output-p/trans4.pm");
    test_matrix("../../../output-p/west0989.pm");
}
