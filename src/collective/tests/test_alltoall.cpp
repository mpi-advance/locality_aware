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

int main(int argc, char** argv)
{
#define LOCAL_COMM_PPN4
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp=RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //


TEST(RandomCommTest, TestsInTests)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Test Integer Alltoall
    int max_i = 10;
    int max_s = pow(2, max_i);
    srand(time(NULL));
    std::vector<int> local_data(max_s*num_procs);

    std::vector<int> std_alltoall(max_s*num_procs);
    std::vector<int> mpix_alltoall(max_s*num_procs);

    MPIX_Comm* xcomm;
    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);
    update_locality(xcomm, 20);
    MPIX_Comm_leader_init(xcomm, 4);

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);

        // Will only be clean for up to double digit process counts
        for (int j = 0; j < num_procs; j++)
            for (int k = 0; k < s; k++)
                local_data[j*s + k] = rank*10000 + j*100 + k;

        // Standard Alltoall
        PMPI_Alltoall(local_data.data(), 
                s,
                MPI_INT, 
                std_alltoall.data(), 
                s, 
                MPI_INT,
                MPI_COMM_WORLD);

        std::fill(mpix_alltoall.begin(), mpix_alltoall.end(), 0);
        MPIX_Alltoall(local_data.data(), 
                s, 
                MPI_INT,
                mpix_alltoall.data(), 
                s, 
                MPI_INT,
                xcomm);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_alltoall[j], mpix_alltoall[j]);

        std::fill(mpix_alltoall.begin(), mpix_alltoall.end(), 0);
        alltoall_pairwise(local_data.data(), 
                s, 
                MPI_INT,
                mpix_alltoall.data(), 
                s, 
                MPI_INT,
                xcomm);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_alltoall[j], mpix_alltoall[j]);

        std::fill(mpix_alltoall.begin(), mpix_alltoall.end(), 0);
        alltoall_nonblocking(local_data.data(), 
                s, 
                MPI_INT,
                mpix_alltoall.data(), 
                s, 
                MPI_INT,
                xcomm);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_alltoall[j], mpix_alltoall[j]);

        alltoall_hierarchical(local_data.data(), 
                s, 
                MPI_INT,
                mpix_alltoall.data(), 
                s, 
                MPI_INT,
                xcomm);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_alltoall[j], mpix_alltoall[j]);

        std::fill(mpix_alltoall.begin(), mpix_alltoall.end(), 0);
        alltoall_multileader(local_data.data(), 
                s, 
                MPI_INT,
                mpix_alltoall.data(), 
                s, 
                MPI_INT,
                xcomm);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_alltoall[j], mpix_alltoall[j]);


        std::fill(mpix_alltoall.begin(), mpix_alltoall.end(), 0);
        alltoall_node_aware(local_data.data(), 
                s, 
                MPI_INT,
                mpix_alltoall.data(), 
                s, 
                MPI_INT,
                xcomm);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_alltoall[j], mpix_alltoall[j]);


        std::fill(mpix_alltoall.begin(), mpix_alltoall.end(), 0);
        alltoall_locality_aware(local_data.data(), 
                s, 
                MPI_INT,
                mpix_alltoall.data(), 
                s, 
                MPI_INT,
                xcomm);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_alltoall[j], mpix_alltoall[j]);


        std::fill(mpix_alltoall.begin(), mpix_alltoall.end(), 0);
        alltoall_multileader_locality(local_data.data(), 
                s, 
                MPI_INT,
                mpix_alltoall.data(), 
                s, 
                MPI_INT,
                xcomm);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_alltoall[j], mpix_alltoall[j]);

break;
/*
    alltoall_hierarchical_nb(local_data.data(), 
                s, 
                MPI_INT,
                mpix_alltoall.data(), 
                s, 
                MPI_INT,
                xcomm);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_alltoall[j], mpix_alltoall[j]);

        std::fill(mpix_alltoall.begin(), mpix_alltoall.end(), 0);
        alltoall_multileader_nb(local_data.data(), 
                s, 
                MPI_INT,
                mpix_alltoall.data(), 
                s, 
                MPI_INT,
                xcomm);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_alltoall[j], mpix_alltoall[j]);


        std::fill(mpix_alltoall.begin(), mpix_alltoall.end(), 0);
        alltoall_node_aware_nb(local_data.data(), 
                s, 
                MPI_INT,
                mpix_alltoall.data(), 
                s, 
                MPI_INT,
                xcomm);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_alltoall[j], mpix_alltoall[j]);


        std::fill(mpix_alltoall.begin(), mpix_alltoall.end(), 0);
        alltoall_locality_aware_nb(local_data.data(), 
                s, 
                MPI_INT,
                mpix_alltoall.data(), 
                s, 
                MPI_INT,
                xcomm);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_alltoall[j], mpix_alltoall[j]);


        std::fill(mpix_alltoall.begin(), mpix_alltoall.end(), 0);
        alltoall_multileader_locality_nb(local_data.data(), 
                s, 
                MPI_INT,
                mpix_alltoall.data(), 
                s, 
                MPI_INT,
                xcomm);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_alltoall[j], mpix_alltoall[j]);

*/
    }

    MPIX_Comm_free(&xcomm);
}


