#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>

#include <iostream>
#include <set>
#include <vector>

#include "locality_aware.h"

void compare_results(std::vector<int> pmpi, std::vector<int>& mpil, int s)
{
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    for (int j = 0; j < s*num_procs; j++)
    {
        if (pmpi[j] != mpil[j]) 
        {
            fprintf(stderr,
                    "MPIL Allgather != PMPI, position %d, pmpi %d, mpil %d\n",
                    j,
                    pmpi[j],
                    mpil[j]);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPIL_Init(MPI_COMM_WORLD);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Test Integer Allgather
    int max_i = 10;
    int max_s = pow(2, max_i);
    srand(time(NULL));
    std::vector<int> local_data(max_s*num_procs);

    std::vector<int> pmpi_data(max_s*num_procs);
    std::vector<int> mpil_data(max_s*num_procs);

    MPIL_Comm* mpil_comm;
    MPIL_Comm_init(&mpil_comm, MPI_COMM_WORLD);

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);

        // Will only be clean for up to double digit process counts
        for (int k = 0; k < s*num_procs; k++)
        {
            local_data[k] = rank * max_s * num_procs + k;
        }

        // PMPI Allgather
        PMPI_Allgather(local_data.data(),
                       s,
                       MPI_INT,
                       pmpi_data.data(),
                       s,
                       MPI_INT,
                       MPI_COMM_WORLD);

        // Default MPIL Allgather
        std::fill(mpil_data.begin(), mpil_data.end(), 0);
        MPIL_Allgather(local_data.data(),
                        s,
                        MPI_INT,
                        mpil_data.data(),
                        s,
                        MPI_INT,
                        mpil_comm);
        compare_results(pmpi_data, mpil_data, s);

        // RING Allgather
        MPIL_Set_allgather_algorithm(ALLGATHER_RING);
        std::fill(mpil_data.begin(), mpil_data.end(), 0);
        MPIL_Allgather(local_data.data(),
                        s,
                        MPI_INT,
                        mpil_data.data(),
                        s,
                        MPI_INT,
                        mpil_comm);
        compare_results(pmpi_data, mpil_data, s);

        // BRUCK Allgather
        MPIL_Set_allgather_algorithm(ALLGATHER_BRUCK);
        std::fill(mpil_data.begin(), mpil_data.end(), 0);
        MPIL_Allgather(local_data.data(),
                        s,
                        MPI_INT,
                        mpil_data.data(),
                        s,
                        MPI_INT,
                        mpil_comm);
        compare_results(pmpi_data, mpil_data, s);
        

        
        // PMPI Allgather
        MPIL_Set_allgather_algorithm(ALLGATHER_PMPI);
        std::fill(mpil_data.begin(), mpil_data.end(), 0);
        MPIL_Allgather(local_data.data(),
                        s,
                        MPI_INT,
                        mpil_data.data(),
                        s,
                        MPI_INT,
                        mpil_comm);
        compare_results(pmpi_data, mpil_data, s);
    }

    MPIL_Comm_free(&mpil_comm);

    MPIL_Finalize();
    MPI_Finalize();
    return 0;
}  // end of main() //
