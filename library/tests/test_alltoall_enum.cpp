#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>

#include <iostream>
#include <set>
#include <vector>

// #include "communicator/MPIL_Comm.h"
#include "locality_aware.h"

void compare_alltoall_results(std::vector<int>& pmpi, std::vector<int>& mpil, int s)
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
    int max_i = 10;
    int max_s = pow(2, max_i);
    srand(time(NULL));
    std::vector<int> local_data(max_s * num_procs);

    std::vector<int> pmpi_alltoall(max_s * num_procs);
    std::vector<int> mpil_alltoall(max_s * num_procs);

    MPIL_Comm* locality_comm;
    MPIL_Comm_init(&locality_comm, MPI_COMM_WORLD);
    MPIL_Comm_update_locality(locality_comm, 4);

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
        PMPI_Alltoall(local_data.data(),
                      s,
                      MPI_INT,
                      pmpi_alltoall.data(),
                      s,
                      MPI_INT,
                      MPI_COMM_WORLD);

        mpil_alltoall_implementation = ALLTOALL_PAIRWISE;
        std::fill(mpil_alltoall.begin(), mpil_alltoall.end(), 0);
        MPIL_Alltoall(local_data.data(),
                      s,
                      MPI_INT,
                      mpil_alltoall.data(),
                      s,
                      MPI_INT,
                      locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpil_alltoall, s);

        mpil_alltoall_implementation = ALLTOALL_NONBLOCKING;
        std::fill(mpil_alltoall.begin(), mpil_alltoall.end(), 0);
        MPIL_Alltoall(local_data.data(),
                      s,
                      MPI_INT,
                      mpil_alltoall.data(),
                      s,
                      MPI_INT,
                      locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpil_alltoall, s);

        mpil_alltoall_implementation = ALLTOALL_HIERARCHICAL_PAIRWISE;
        std::fill(mpil_alltoall.begin(), mpil_alltoall.end(), 0);
        MPIL_Alltoall(local_data.data(),
                      s,
                      MPI_INT,
                      mpil_alltoall.data(),
                      s,
                      MPI_INT,
                      locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpil_alltoall, s);

        mpil_alltoall_implementation = ALLTOALL_HIERARCHICAL_NONBLOCKING;
        std::fill(mpil_alltoall.begin(), mpil_alltoall.end(), 0);
        MPIL_Alltoall(local_data.data(),
                      s,
                      MPI_INT,
                      mpil_alltoall.data(),
                      s,
                      MPI_INT,
                      locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpil_alltoall, s);

        mpil_alltoall_implementation = ALLTOALL_MULTILEADER_PAIRWISE;
        std::fill(mpil_alltoall.begin(), mpil_alltoall.end(), 0);
        MPIL_Alltoall(local_data.data(),
                      s,
                      MPI_INT,
                      mpil_alltoall.data(),
                      s,
                      MPI_INT,
                      locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpil_alltoall, s);

        mpil_alltoall_implementation = ALLTOALL_MULTILEADER_NONBLOCKING;
        std::fill(mpil_alltoall.begin(), mpil_alltoall.end(), 0);
        MPIL_Alltoall(local_data.data(),
                      s,
                      MPI_INT,
                      mpil_alltoall.data(),
                      s,
                      MPI_INT,
                      locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpil_alltoall, s);

        mpil_alltoall_implementation = ALLTOALL_NODE_AWARE_PAIRWISE;
        std::fill(mpil_alltoall.begin(), mpil_alltoall.end(), 0);
        MPIL_Alltoall(local_data.data(),
                      s,
                      MPI_INT,
                      mpil_alltoall.data(),
                      s,
                      MPI_INT,
                      locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpil_alltoall, s);

        mpil_alltoall_implementation = ALLTOALL_NODE_AWARE_NONBLOCKING;
        std::fill(mpil_alltoall.begin(), mpil_alltoall.end(), 0);
        MPIL_Alltoall(local_data.data(),
                      s,
                      MPI_INT,
                      mpil_alltoall.data(),
                      s,
                      MPI_INT,
                      locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpil_alltoall, s);

        mpil_alltoall_implementation = ALLTOALL_LOCALITY_AWARE_PAIRWISE;
        std::fill(mpil_alltoall.begin(), mpil_alltoall.end(), 0);
        MPIL_Alltoall(local_data.data(),
                      s,
                      MPI_INT,
                      mpil_alltoall.data(),
                      s,
                      MPI_INT,
                      locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpil_alltoall, s);

        mpil_alltoall_implementation = ALLTOALL_LOCALITY_AWARE_NONBLOCKING;
        std::fill(mpil_alltoall.begin(), mpil_alltoall.end(), 0);
        MPIL_Alltoall(local_data.data(),
                      s,
                      MPI_INT,
                      mpil_alltoall.data(),
                      s,
                      MPI_INT,
                      locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpil_alltoall, s);

        mpil_alltoall_implementation = ALLTOALL_MULTILEADER_LOCALITY_PAIRWISE;
        std::fill(mpil_alltoall.begin(), mpil_alltoall.end(), 0);
        MPIL_Alltoall(local_data.data(),
                      s,
                      MPI_INT,
                      mpil_alltoall.data(),
                      s,
                      MPI_INT,
                      locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpil_alltoall, s);

        mpil_alltoall_implementation = ALLTOALL_MULTILEADER_LOCALITY_NONBLOCKING;
        std::fill(mpil_alltoall.begin(), mpil_alltoall.end(), 0);
        MPIL_Alltoall(local_data.data(),
                      s,
                      MPI_INT,
                      mpil_alltoall.data(),
                      s,
                      MPI_INT,
                      locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpil_alltoall, s);

        mpil_alltoall_implementation = ALLTOALL_PMPI;
        std::fill(mpil_alltoall.begin(), mpil_alltoall.end(), 0);
        MPIL_Alltoall(local_data.data(),
                      s,
                      MPI_INT,
                      mpil_alltoall.data(),
                      s,
                      MPI_INT,
                      locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpil_alltoall, s);
    }

    MPIL_Comm_free(&locality_comm);

    MPI_Finalize();
    return 0;
}  // end of main() //
