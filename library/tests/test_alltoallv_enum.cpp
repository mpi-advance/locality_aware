#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>

#include <iostream>
#include <set>
#include <vector>

#include "locality_aware.h"

void compare_alltoallv_results(std::vector<int>& pmpi, std::vector<int>& mpil, int s)
{
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    for (int j = 0; j < s * num_procs; j++)
    {
        if (pmpi[j] != mpil[j])
        {
            fprintf(stderr,
                    "MPIL Alltoallv != PMPI, position %d, pmpi %d, mpil %d\n",
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

    // Test Integer Alltoall
    int max_i = 10;
    int max_s = pow(2, max_i);
    srand(time(NULL));
    std::vector<int> local_data(max_s * num_procs);

    std::vector<int> pmpi_alltoallv(max_s * num_procs);
    std::vector<int> mpil_alltoallv(max_s * num_procs);

    std::vector<int> sizes(num_procs);
    std::vector<int> displs(num_procs + 1);

    MPIL_Comm* xcomm;
    MPIL_Comm_init(&xcomm, MPI_COMM_WORLD);
    MPIL_Comm_update_locality(xcomm, 4);

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);

        // Will only be clean for up to double digit process counts
        displs[0] = 0;
        for (int j = 0; j < num_procs; j++)
        {
            for (int k = 0; k < s; k++)
            {
                local_data[j * s + k] = rank * 10000 + j * 100 + k;
            }
            sizes[j]      = s;
            displs[j + 1] = displs[j] + s;
        }

        PMPI_Alltoallv(local_data.data(),
                       sizes.data(),
                       displs.data(),
                       MPI_INT,
                       pmpi_alltoallv.data(),
                       sizes.data(),
                       displs.data(),
                       MPI_INT,
                       MPI_COMM_WORLD);

        mpil_alltoallv_implementation = ALLTOALLV_PAIRWISE;
        std::fill(mpil_alltoallv.begin(), mpil_alltoallv.end(), 0);
        MPIL_Alltoallv(local_data.data(),
                       sizes.data(),
                       displs.data(),
                       MPI_INT,
                       mpil_alltoallv.data(),
                       sizes.data(),
                       displs.data(),
                       MPI_INT,
                       xcomm);
        compare_alltoallv_results(pmpi_alltoallv, mpil_alltoallv, s);

        mpil_alltoallv_implementation = ALLTOALLV_NONBLOCKING;
        std::fill(mpil_alltoallv.begin(), mpil_alltoallv.end(), 0);
        MPIL_Alltoallv(local_data.data(),
                       sizes.data(),
                       displs.data(),
                       MPI_INT,
                       mpil_alltoallv.data(),
                       sizes.data(),
                       displs.data(),
                       MPI_INT,
                       xcomm);
        compare_alltoallv_results(pmpi_alltoallv, mpil_alltoallv, s);

        mpil_alltoallv_implementation = ALLTOALLV_BATCH;
        std::fill(mpil_alltoallv.begin(), mpil_alltoallv.end(), 0);
        MPIL_Alltoallv(local_data.data(),
                       sizes.data(),
                       displs.data(),
                       MPI_INT,
                       mpil_alltoallv.data(),
                       sizes.data(),
                       displs.data(),
                       MPI_INT,
                       xcomm);
        compare_alltoallv_results(pmpi_alltoallv, mpil_alltoallv, s);

        mpil_alltoallv_implementation = ALLTOALLV_BATCH_ASYNC;
        std::fill(mpil_alltoallv.begin(), mpil_alltoallv.end(), 0);
        MPIL_Alltoallv(local_data.data(),
                       sizes.data(),
                       displs.data(),
                       MPI_INT,
                       mpil_alltoallv.data(),
                       sizes.data(),
                       displs.data(),
                       MPI_INT,
                       xcomm);
        compare_alltoallv_results(pmpi_alltoallv, mpil_alltoallv, s);

        mpil_alltoallv_implementation = ALLTOALLV_PMPI;
        std::fill(mpil_alltoallv.begin(), mpil_alltoallv.end(), 0);
        MPIL_Alltoallv(local_data.data(),
                       sizes.data(),
                       displs.data(),
                       MPI_INT,
                       mpil_alltoallv.data(),
                       sizes.data(),
                       displs.data(),
                       MPI_INT,
                       xcomm);
        compare_alltoallv_results(pmpi_alltoallv, mpil_alltoallv, s);
    }

    MPIL_Comm_free(&xcomm);

    MPIL_Finalize();
    MPI_Finalize();
    return 0;
}  // end of main() //
