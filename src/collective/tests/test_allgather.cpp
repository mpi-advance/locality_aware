#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>

void compare_allgather_results(std::vector<int>& pmpi, std::vector<int>& mpix, int s)
{
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    for (int j = 0; j < s*num_procs; j++)
    {
        if (pmpi[j] != mpix[j])
        {
            fprintf(stderr, "MPIX Allgather != PMPI, position %d, pmpi %d, mpix %d\n", 
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

    MPIX_Comm* xcomm;
    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);
    update_locality(xcomm, 4);

    // Test Integer Alltoall
    int max_i = 10;
    int max_s = pow(2, max_i);
    srand(time(NULL));
    std::vector<int> local_data(max_s);

    std::vector<int> mpix_allgather(max_s*num_procs);
    std::vector<int> pmpi_allgather(max_s*num_procs);

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
                pmpi_allgather.data(), 
                s, 
                MPI_INT,
                MPI_COMM_WORLD);

        // Bruck Allgather 
        std::fill(mpix_allgather.begin(), mpix_allgather.end(), 0);
        allgather_bruck(local_data.data(), 
                s, 
                MPI_INT,
                mpix_allgather.data(), 
                s, 
                MPI_INT,
                MPI_COMM_WORLD);
        compare_allgather_results(pmpi_allgather, mpix_allgather, s);


        // P2P Allgather 
        std::fill(mpix_allgather.begin(), mpix_allgather.end(), 0);
        allgather_p2p(local_data.data(), 
                s, 
                MPI_INT,
                mpix_allgather.data(), 
                s, 
                MPI_INT,
                MPI_COMM_WORLD);
        compare_allgather_results(pmpi_allgather, mpix_allgather, s);

        // Ring Allgather 
        std::fill(mpix_allgather.begin(), mpix_allgather.end(), 0);
        allgather_ring(local_data.data(), 
                s, 
                MPI_INT,
                mpix_allgather.data(), 
                s, 
                MPI_INT,
                MPI_COMM_WORLD);
        compare_allgather_results(pmpi_allgather, mpix_allgather, s);

        // Locality P2P Allgather 
        std::fill(mpix_allgather.begin(), mpix_allgather.end(), 0);
        allgather_loc_p2p(local_data.data(), 
                s, 
                MPI_INT,
                mpix_allgather.data(), 
                s, 
                MPI_INT,
                xcomm);
        compare_allgather_results(pmpi_allgather, mpix_allgather, s);

        // Locality Bruck Allgather 
        std::fill(mpix_allgather.begin(), mpix_allgather.end(), 0);
        allgather_loc_bruck(local_data.data(), 
                s, 
                MPI_INT,
                mpix_allgather.data(), 
                s, 
                MPI_INT,
                xcomm);
        compare_allgather_results(pmpi_allgather, mpix_allgather, s);

        // Locality Ring Allgather 
        std::fill(mpix_allgather.begin(), mpix_allgather.end(), 0);
        allgather_loc_ring(local_data.data(), 
                s, 
                MPI_INT,
                mpix_allgather.data(), 
                s, 
                MPI_INT,
                xcomm);
        compare_allgather_results(pmpi_allgather, mpix_allgather, s);

        // Hierarchical Bruck Allgather 
        std::fill(mpix_allgather.begin(), mpix_allgather.end(), 0);
        allgather_hier_bruck(local_data.data(), 
                s, 
                MPI_INT,
                mpix_allgather.data(), 
                s, 
                MPI_INT,
                xcomm);
        compare_allgather_results(pmpi_allgather, mpix_allgather, s);

        // Hierarchical (MULT) Bruck Allgather 
        std::fill(mpix_allgather.begin(), mpix_allgather.end(), 0);
        allgather_mult_hier_bruck(local_data.data(), 
                s, 
                MPI_INT,
                mpix_allgather.data(), 
                s, 
                MPI_INT,
                xcomm);
        compare_allgather_results(pmpi_allgather, mpix_allgather, s);
    }

    MPIX_Comm_free(&xcomm);

    MPI_Finalize();

    return 0;
} 


