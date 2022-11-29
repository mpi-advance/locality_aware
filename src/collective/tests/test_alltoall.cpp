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
    std::vector<int> loc_p2p_alltoall(max_s*num_procs);
    std::vector<int> bruck_alltoall(max_s*num_procs);

    MPIX_Comm* locality_comm;
    MPIX_Comm_init(&locality_comm, MPI_COMM_WORLD);

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);

        // Will only be clean for up to double digit process counts
        for (int i = 0; i < num_procs; i++)
            for (int j = 0; j < s; j++)
                local_data[i*s + j] = rank*10000 + i*100 + j;

        // Standard Alltoall
        PMPI_Alltoall(local_data.data(), 
                s,
                MPI_INT, 
                std_alltoall.data(), 
                s, 
                MPI_INT,
                MPI_COMM_WORLD);


        // Locality-Aware P2P Alltoall 
        MPI_Alltoall(local_data.data(), 
                s,
                MPI_INT, 
                loc_p2p_alltoall.data(), 
                s, 
                MPI_INT,
                MPI_COMM_WORLD);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_alltoall[j], loc_p2p_alltoall[j]);

        MPIX_Alltoall(local_data.data(), 
                s, 
                MPI_INT,
                loc_p2p_alltoall.data(), 
                s, 
                MPI_INT,
                locality_comm);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_alltoall[j], loc_p2p_alltoall[j]);

        alltoall_bruck(local_data.data(), 
                s, 
                MPI_INT,
                bruck_alltoall.data(), 
                s, 
                MPI_INT,
                MPI_COMM_WORLD);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_alltoall[j], bruck_alltoall[j]);
    }

    MPIX_Comm_free(locality_comm);
}


