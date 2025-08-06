#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>

void compare_alltoallv_results(std::vector<int>& pmpi, std::vector<int>& mpix, int s)
{
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    for (int j = 0; j < s*num_procs; j++)
    {
        if (pmpi[j] != mpix[j])
        {
            fprintf(stderr, "MPIX Alltoallv != PMPI, position %d, pmpi %d, mpix %d\n", 
                    j, pmpi[j], mpix[j]);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Test Integer Alltoall
    int max_i = 10;
    int max_s = pow(2, max_i);
    srand(time(NULL));
    std::vector<int> local_data(max_s*num_procs);

    std::vector<int> pmpi_alltoallv(max_s*num_procs);
    std::vector<int> mpix_alltoallv(max_s*num_procs);

    std::vector<int> sendcounts(num_procs);
    std::vector<int> recvcounts(num_procs);
    std::vector<int> sdispls(num_procs+1);
    std::vector<int> rdispls(num_procs+1);

    MPIX_Comm* xcomm;
    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);
    update_locality(xcomm, 4);

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);

        // Will only be clean for up to double digit process counts
        sdispls[0] = 0;
        rdispls[0] = 0;
        for (int j = 0; j < num_procs; j++)
        {
            for (int k = 0; k < s; k++)
                local_data[j*s + k] = rank*10000 + j*100 + k;
            sendcounts[j] = s;
            sdispls[j+1] = sdispls[j] + s;
            recvcounts[j] = s;
            rdispls[j+1] = rdispls[j] + s;
        }


        PMPI_Alltoallv(local_data.data(), 
                sendcounts.data(),
                sdispls.data(),
                MPI_INT, 
                pmpi_alltoallv.data(), 
                recvcounts.data(),
                rdispls.data(),
                MPI_INT,
                MPI_COMM_WORLD);

        mpix_alltoallv_implementation = ALLTOALLV_PAIRWISE;
        std::fill(mpix_alltoallv.begin(), mpix_alltoallv.end(), 0);
        MPIX_Alltoallv(local_data.data(), 
                sendcounts.data(),
                sdispls.data(),
                MPI_INT, 
                mpix_alltoallv.data(), 
                recvcounts.data(),
                rdispls.data(),
                MPI_INT,
                xcomm);
        compare_alltoallv_results(pmpi_alltoallv, mpix_alltoallv, s);

        mpix_alltoallv_implementation = ALLTOALLV_NONBLOCKING;
        std::fill(mpix_alltoallv.begin(), mpix_alltoallv.end(), 0);
        MPIX_Alltoallv(local_data.data(), 
                sendcounts.data(),
                sdispls.data(),
                MPI_INT, 
                mpix_alltoallv.data(), 
                recvcounts.data(),
                rdispls.data(),
                MPI_INT,
                xcomm);
        compare_alltoallv_results(pmpi_alltoallv, mpix_alltoallv, s);

        mpix_alltoallv_implementation = ALLTOALLV_BATCH;
        std::fill(mpix_alltoallv.begin(), mpix_alltoallv.end(), 0);
        MPIX_Alltoallv(local_data.data(), 
                sendcounts.data(),
                sdispls.data(),
                MPI_INT, 
                mpix_alltoallv.data(), 
                recvcounts.data(),
                rdispls.data(),
                MPI_INT,
                xcomm);
        compare_alltoallv_results(pmpi_alltoallv, mpix_alltoallv, s);

        mpix_alltoallv_implementation = ALLTOALLV_BATCH_ASYNC;
        std::fill(mpix_alltoallv.begin(), mpix_alltoallv.end(), 0);
        MPIX_Alltoallv(local_data.data(), 
                sendcounts.data(),
                sdispls.data(),
                MPI_INT, 
                mpix_alltoallv.data(), 
                recvcounts.data(),
                rdispls.data(),
                MPI_INT,
                xcomm);
        compare_alltoallv_results(pmpi_alltoallv, mpix_alltoallv, s);

        mpix_alltoallv_implementation = ALLTOALLV_PMPI;
        std::fill(mpix_alltoallv.begin(), mpix_alltoallv.end(), 0);
        MPIX_Alltoallv(local_data.data(), 
                sendcounts.data(),
                sdispls.data(),
                MPI_INT, 
                mpix_alltoallv.data(), 
                recvcounts.data(),
                rdispls.data(),
                MPI_INT,
                xcomm);
        compare_alltoallv_results(pmpi_alltoallv, mpix_alltoallv, s);

    }


    MPIX_Comm_free(&xcomm);


    MPI_Finalize();
    return 0;
} // end of main() //



