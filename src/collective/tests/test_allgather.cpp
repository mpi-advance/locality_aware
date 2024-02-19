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

    MPIX_Comm* locality_comm;
    MPIX_Comm_init(&locality_comm, MPI_COMM_WORLD);
    update_locality(locality_comm, 4);

    // Test Integer Alltoall
    int max_i = 10;
    int max_s = pow(2, max_i);
    srand(time(NULL));
    std::vector<int> local_data(max_s);

    std::vector<int> std_allgather(max_s*num_procs);
    std::vector<int> bruck_allgather(max_s*num_procs);
    std::vector<int> p2p_allgather(max_s*num_procs);
    std::vector<int> ring_allgather(max_s*num_procs);
    std::vector<int> loc_p2p_allgather(max_s*num_procs);
    std::vector<int> loc_bruck_allgather(max_s*num_procs);
    std::vector<int> loc_ring_allgather(max_s*num_procs);
    std::vector<int> hier_bruck_allgather(max_s*num_procs);
    std::vector<int> mult_hier_bruck_allgather(max_s*num_procs);


    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);

        // Will only be clean for up to double digit process counts
        for (int j = 0; j < s; j++)
            local_data[j] = rank*100 + j;

        // Standard Allgather
        PMPI_Allgather(local_data.data(), 
                s,
                MPI_INT, 
                std_allgather.data(), 
                s, 
                MPI_INT,
                MPI_COMM_WORLD);

        // Bruck Allgather 
        allgather_bruck(local_data.data(), 
                s, 
                MPI_INT,
                bruck_allgather.data(), 
                s, 
                MPI_INT,
                MPI_COMM_WORLD);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_allgather[j], bruck_allgather[j]);


        // P2P Allgather 
        allgather_p2p(local_data.data(), 
                s, 
                MPI_INT,
                p2p_allgather.data(), 
                s, 
                MPI_INT,
                MPI_COMM_WORLD);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_allgather[j], p2p_allgather[j]);


        // Ring Allgather 
        allgather_ring(local_data.data(), 
                s, 
                MPI_INT,
                ring_allgather.data(), 
                s, 
                MPI_INT,
                MPI_COMM_WORLD);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_allgather[j], ring_allgather[j]);

        // Locality P2P Allgather 
        allgather_loc_p2p(local_data.data(), 
                s, 
                MPI_INT,
                loc_p2p_allgather.data(), 
                s, 
                MPI_INT,
                locality_comm);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_allgather[j], loc_p2p_allgather[j]);


        // Locality Bruck Allgather 
        allgather_loc_bruck(local_data.data(), 
                s, 
                MPI_INT,
                loc_bruck_allgather.data(), 
                s, 
                MPI_INT,
                locality_comm);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_allgather[j], loc_bruck_allgather[j]);

        // Locality Ring Allgather 
        allgather_loc_ring(local_data.data(), 
                s, 
                MPI_INT,
                loc_ring_allgather.data(), 
                s, 
                MPI_INT,
                locality_comm);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_allgather[j], loc_ring_allgather[j]);

        // Hierarchical Bruck Allgather 
        allgather_hier_bruck(local_data.data(), 
                s, 
                MPI_INT,
                hier_bruck_allgather.data(), 
                s, 
                MPI_INT,
                locality_comm);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_allgather[j], hier_bruck_allgather[j]);

        // Hierarchical (MULT) Bruck Allgather 
        allgather_mult_hier_bruck(local_data.data(), 
                s, 
                MPI_INT,
                mult_hier_bruck_allgather.data(), 
                s, 
                MPI_INT,
                locality_comm);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_allgather[j], mult_hier_bruck_allgather[j]);
    }

    MPIX_Comm_free(&locality_comm);
}


