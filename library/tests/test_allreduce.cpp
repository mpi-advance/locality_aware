#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>

#include <iostream>
#include <set>
#include <vector>

#include "locality_aware.h"

void compare_allreduce_results(std::vector<int> pmpi, std::vector<int>& mpil, int s)
{
    for (int j = 0; j < s; j++)
    {
        if (pmpi[j] != mpil[j]) // integer allreduce, no rounding error
        {
            fprintf(stderr,
                    "MPIL Allreduce != PMPI, position %d, pmpi %d, mpil %d\n",
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

    std::vector<int> pmpi_allreduce(max_s);
    std::vector<int> mpil_allreduce(max_s);

    MPIL_Comm* locality_comm;
    MPIL_Comm_init(&locality_comm, MPI_COMM_WORLD);

    // Assume 4 ranks per node for testing
    MPIL_Comm_update_locality(locality_comm, 4);

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);

        // Will only be clean for up to double digit process counts
        for (int k = 0; k < s; k++)
        {
            local_data[k] = rank * max_s + k;
        }

        // PMPI Alltoall
        PMPI_Allreduce(local_data.data(),
                       pmpi_allreduce.data(),
                       s,
                       MPI_INT,
                       MPI_SUM,
                       MPI_COMM_WORLD);

        // Default MPIL Allreduce
        std::fill(mpil_allreduce.begin(), mpil_allreduce.end(), 0);
        MPIL_Allreduce(local_data.data(),
                       mpil_allreduce.data(),
                       s, 
                       MPI_INT,
                       MPI_SUM,
                       locality_comm);
        compare_allreduce_results(pmpi_allreduce, mpil_allreduce, s);

        // Recursive-Doubling MPIL Allreduce
        std::fill(mpil_allreduce.begin(), mpil_allreduce.end(), 0);
        MPIL_Set_allreduce_algorithm(ALLREDUCE_RECURSIVE_DOUBLING);
        MPIL_Allreduce(local_data.data(),
                       mpil_allreduce.data(),
                       s, 
                       MPI_INT,
                       MPI_SUM,
                       locality_comm);
        compare_allreduce_results(pmpi_allreduce, mpil_allreduce, s);


        // Dissemination Locality MPIL Allreduce
        std::fill(mpil_allreduce.begin(), mpil_allreduce.end(), 0);
        MPIL_Set_allreduce_algorithm(ALLREDUCE_DISSEMINATION_LOC);
        MPIL_Allreduce(local_data.data(),
                       mpil_allreduce.data(),
                       s, 
                       MPI_INT,
                       MPI_SUM,
                       locality_comm);
        compare_allreduce_results(pmpi_allreduce, mpil_allreduce, s);

        // Dissemination Locality (Multileader) MPIL Allreduce
        std::fill(mpil_allreduce.begin(), mpil_allreduce.end(), 0);
        MPIL_Set_allreduce_algorithm(ALLREDUCE_DISSEMINATION_ML);
        MPIL_Allreduce(local_data.data(),
                       mpil_allreduce.data(),
                       s, 
                       MPI_INT,
                       MPI_SUM,
                       locality_comm);
        compare_allreduce_results(pmpi_allreduce, mpil_allreduce, s);

        // Dissemination RADIX MPIL Allreduce
        std::fill(mpil_allreduce.begin(), mpil_allreduce.end(), 0);
        MPIL_Set_allreduce_algorithm(ALLREDUCE_DISSEMINATION_RADIX);
        MPIL_Allreduce(local_data.data(),
                       mpil_allreduce.data(),
                       s, 
                       MPI_INT,
                       MPI_SUM,
                       locality_comm);
        compare_allreduce_results(pmpi_allreduce, mpil_allreduce, s);
    }

    MPIL_Comm_free(&locality_comm);

    MPI_Finalize();
    return 0;
}  // end of main() //
