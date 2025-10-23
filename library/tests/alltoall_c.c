#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "locality_aware.h"

void compare_alltoall_results(int* pmpi, int* mpil, int s)
{
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    for (int j = 0; j < s * num_procs; j++)
    {
        if (pmpi[j] != mpil[j])
        {
            fprintf(stderr,
                    "MPIL Alltoall != PMPI, position %d, pmpi %d, mpil %d\n",
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

    // Test Integer Alltoall
    int max_i          = 10;
    int max_s          = pow(2, max_i);
    int elements       = max_s * num_procs;
    int* local_data    = malloc(elements * sizeof(int));
    int* pmpi_alltoall = malloc(elements * sizeof(int));
    int* mpil_alltoall = malloc(elements * sizeof(int));

    MPIL_Comm* locality_comm;
    MPIL_Comm_init(&locality_comm, MPI_COMM_WORLD);
    update_locality(locality_comm, 4);

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);

        // Will only be clean for up to double digit process counts
        for (int j = 0; j < num_procs; j++)
        {
            for (int k = 0; k < s; k++)
            {
                local_data[j * s + k] = rank * 10000 + j * 100 + k;
            }
        }

        // Standard Alltoall
        PMPI_Alltoall(local_data, s, MPI_INT, pmpi_alltoall, s, MPI_INT, MPI_COMM_WORLD);

        // Locality-Aware Pairwise Alltoall
        for (int i = 0; i < elements; i++)
        {
            mpil_alltoall[i] = 0;
        }

        MPIL_Alltoall(local_data, s, MPI_INT, mpil_alltoall, s, MPI_INT, locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpil_alltoall, s);
    }

    MPIL_Comm_free(&locality_comm);

    free(local_data);
    free(pmpi_alltoall);
    free(mpil_alltoall);

    MPI_Finalize();
    return 0;
}  // end of main() //
