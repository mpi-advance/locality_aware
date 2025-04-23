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

    std::vector<int> std_alltoallv(max_s*num_procs);
    std::vector<int> pairwise_alltoallv(max_s*num_procs);
    std::vector<int> loc_pairwise_alltoallv(max_s*num_procs);

    std::vector<int> sizes(num_procs);
    std::vector<int> displs(num_procs+1);

    MPIX_Comm* locality_comm;
    MPIX_Comm_init(&locality_comm, MPI_COMM_WORLD);
    update_locality(locality_comm, 4);

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);

        // Will only be clean for up to double digit process counts
        displs[0] = 0;
        for (int j = 0; j < num_procs; j++)
        {
            for (int k = 0; k < s; k++)
                local_data[j*s + k] = rank*10000 + j*100 + k;
            sizes[j] = s;
            displs[j+1] = displs[j] + s;
        }


        // Standard Alltoall
        PMPI_Alltoallv(local_data.data(), 
                sizes.data(),
                displs.data(),
                MPI_INT, 
                std_alltoallv.data(), 
                sizes.data(),
                displs.data(),
                MPI_INT,
                MPI_COMM_WORLD);

        // Locality-Aware P2P Alltoallv
        MPI_Alltoallv(local_data.data(), 
                sizes.data(),
                displs.data(),
                MPI_INT, 
                pairwise_alltoallv.data(), 
                sizes.data(),
                displs.data(),
                MPI_INT,
                MPI_COMM_WORLD);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_alltoallv[j], pairwise_alltoallv[j]);

        MPIX_Alltoallv(local_data.data(), 
                sizes.data(),
                displs.data(),
                MPI_INT, 
                loc_pairwise_alltoallv.data(), 
                sizes.data(),
                displs.data(),
                MPI_INT,
                locality_comm);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_alltoallv[j], loc_pairwise_alltoallv[j]);
    }

    MPIX_Comm_free(&locality_comm);
}


