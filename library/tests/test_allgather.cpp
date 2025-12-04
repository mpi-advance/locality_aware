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

    for (int j = 0; j < s * num_procs; j++)
    {
        if (pmpi[j] != mpil[j]) // integer allreduce, no rounding error
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

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Test Integer Allreduce
    int max_i = 10;
    int max_s = pow(2, max_i);
    srand(time(NULL));
    std::vector<int> local_data(max_s);

    std::vector<int> pmpi(max_s*num_procs);
    std::vector<int> mpil(max_s*num_procs);

    // Assume 4 ranks per node for testing
    MPIL_Comm* xcomm;
    MPIL_Comm_init(&xcomm, MPI_COMM_WORLD);
    MPIL_Comm_update_locality(xcomm, 4);

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);

        // Will only be clean for up to double digit process counts
        for (int k = 0; k < s; k++)
        {
            local_data[k] = rank * max_s + k;
        }

        // PMPI Allgather
        PMPI_Allgather(local_data.data(), s, MPI_INT,
                pmpi.data(), s, MPI_INT, MPI_COMM_WORLD);

        // Default MPIL Allreduce
        std::fill(mpil.begin(), mpil.end(), 0);
        MPIL_Allgather(local_data.data(), s, MPI_INT,
                mpil.data(), s, MPI_INT, xcomm);
        compare_results(pmpi, mpil, s);

        // Ring MPIL Allreduce
        MPIL_Set_allgather_algorithm(ALLGATHER_RING);
        std::fill(mpil.begin(), mpil.end(), 0);
        MPIL_Allgather(local_data.data(), s, MPI_INT,
                mpil.data(), s, MPI_INT, xcomm);
        compare_results(pmpi, mpil, s);

        // Bruck MPIL Allreduce
        MPIL_Set_allgather_algorithm(ALLGATHER_RING);
        std::fill(mpil.begin(), mpil.end(), 0);
        MPIL_Allgather(local_data.data(), s, MPI_INT,
                mpil.data(), s, MPI_INT, xcomm);
        compare_results(pmpi, mpil, s);

        // PMPI MPIL Allreduce
        MPIL_Set_allgather_algorithm(ALLGATHER_PMPI);
        std::fill(mpil.begin(), mpil.end(), 0);
        MPIL_Allgather(local_data.data(), s, MPI_INT,
                mpil.data(), s, MPI_INT, xcomm);
        compare_results(pmpi, mpil, s);
    }

    MPIL_Comm_free(&xcomm);

    MPI_Finalize();
    return 0;
}  // end of main() //
