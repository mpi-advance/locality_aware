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
    std::vector<int> std_alltoallv(max_s*num_procs);
    std::vector<int> pairwise_alltoallv(max_s*num_procs);
    std::vector<int> nonblocking_alltoallv(max_s*num_procs);
    std::vector<int> rma_alltoallv(max_s*num_procs);
                         
     std::vector<int> alltoallvint(max_s*num_procs); 
    std::vector<int> sizes(num_procs);
    std::vector<int> displs(num_procs+1);

    MPIX_Comm* xcomm;
    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);
    update_locality(xcomm, 4);

    MPIX_Info* xinfo;
    MPIX_Info_init(&xinfo);

    MPIX_Request* xrequest;
    MPI_Request request;

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);


        // Will only be clean for up to double digit process counts
        displs[0] = 0;
        for (int i = 0; i < num_procs; i++)
        {
            for (int j = 0; j < s; j++)
             local_data[i*s + j] = rank*10000 + i*100 + j;
            sizes[i] = s;
            displs[i+1] = displs[i] + s;
        }



      /*  PMPI_Alltoall_init(local_data.data(), 
                s,
                MPI_INT, 
                std_alltoall.data(), 
                s, 
                MPI_INT,
                MPI_COMM_WORLD, 
                MPI_INFO_NULL,
                &request);
*/





PMPI_Alltoallv(local_data.data(),
                sizes.data(),
                displs.data(),
                MPI_INT,
                std_alltoallv.data(),
                sizes.data(),
                displs.data(),
                MPI_INT,
                MPI_COMM_WORLD);



       alltoallv_nonblocking_init(local_data.data(), 
               
                sizes.data(),
                displs.data(), 
                MPI_INT, 
                nonblocking_alltoallv.data(), 
           
                sizes.data(),
                displs.data(), 
                MPI_INT,
                xcomm,
                 
                xinfo,
                &xrequest);
        MPIX_Start(xrequest);
        MPIX_Wait(xrequest, MPI_STATUS_IGNORE);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_alltoallv[j], nonblocking_alltoallv[j]);
        MPIX_Request_free(xrequest);



alltoallv_init(local_data.data(),

                sizes.data(),
                displs.data(),
                MPI_INT,
                alltoallvint.data(),

                sizes.data(),
                displs.data(),
                MPI_INT,
                xcomm,

                xinfo,
                &xrequest);
        MPIX_Start(xrequest);
        MPIX_Wait(xrequest, MPI_STATUS_IGNORE);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_alltoallv[j], alltoallvint[j]);
        MPIX_Request_free(xrequest);


alltoallv_pairwise_init(local_data.data(),

                sizes.data(),
                displs.data(),
                MPI_INT,
                pairwise_alltoallv.data(),

                sizes.data(),
                displs.data(),
                MPI_INT,
                xcomm,

                xinfo,
                &xrequest);
        MPIX_Start(xrequest);
        MPIX_Wait(xrequest, MPI_STATUS_IGNORE);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_alltoallv[j], pairwise_alltoallv[j]);
        MPIX_Request_free(xrequest);



    alltoallv_rma_init(local_data.data(),

                sizes.data(),
                displs.data(),
                MPI_INT,
                rma_alltoallv.data(),

                sizes.data(),
                displs.data(),
                MPI_INT,
                xcomm,

                xinfo,
                &xrequest);
        MPIX_Start(xrequest);
        MPIX_Wait(xrequest, MPI_STATUS_IGNORE);
        for (int j = 0; j < s*num_procs; j++)
            ASSERT_EQ(std_alltoallv[j], rma_alltoallv[j]);
        MPIX_Request_free(xrequest);



    }


    MPIX_Info_free(&xinfo);
    MPIX_Comm_free(&xcomm);
}


