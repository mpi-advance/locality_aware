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

    MPIL_Info* mpil_info;
    MPIL_Info_init(&mpil_info);

    MPIL_Request* mpil_request;

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

        // Default MPIL Allgather init
        std::fill(mpil_data.begin(), mpil_data.end(), 0);
        MPIL_Allgather_init(local_data.data(),
                            s, 
                            MPI_INT,
                            mpil_data.data(),
                            s, 
                            MPI_INT,
                            mpil_comm,
                            mpil_info,
                            &mpil_request);
        MPIL_Start(mpil_request);
        MPIL_Wait(mpil_request, MPI_STATUS_IGNORE);
        MPIL_Request_free(&mpil_request);
        compare_results(pmpi_data, mpil_data, s);

        // RING MPIL Allgather init
        MPIL_Set_allgather_init_algorithm(ALLGATHER_INIT_RING);
        std::fill(mpil_data.begin(), mpil_data.end(), 0);
        MPIL_Allgather_init(local_data.data(),
                            s, 
                            MPI_INT,
                            mpil_data.data(),
                            s, 
                            MPI_INT,
                            mpil_comm,
                            mpil_info,
                            &mpil_request);
        MPIL_Start(mpil_request);
        MPIL_Wait(mpil_request, MPI_STATUS_IGNORE);
        MPIL_Request_free(&mpil_request);
        compare_results(pmpi_data, mpil_data, s);

        // BRUCK MPIL Allgather init
        MPIL_Set_allgather_init_algorithm(ALLGATHER_INIT_BRUCK);
        std::fill(mpil_data.begin(), mpil_data.end(), 0);
        MPIL_Allgather_init(local_data.data(),
                            s, 
                            MPI_INT,
                            mpil_data.data(),
                            s, 
                            MPI_INT,
                            mpil_comm,
                            mpil_info,
                            &mpil_request);
        MPIL_Start(mpil_request);
        MPIL_Wait(mpil_request, MPI_STATUS_IGNORE);
        MPIL_Request_free(&mpil_request);
        compare_results(pmpi_data, mpil_data, s);

#if defined(MPI4)
        // PMPI MPIL Allgather init
        MPIL_Set_allgather_init_algorithm(ALLGATHER_INIT_PMPI);
        std::fill(mpil_data.begin(), mpil_data.end(), 0);
        MPIL_Allgather_init(local_data.data(),
                            s, 
                            MPI_INT,
                            mpil_data.data(),
                            s, 
                            MPI_INT,
                            mpil_comm,
                            mpil_info,
                            &mpil_request);
        MPIL_Start(mpil_request);
        MPIL_Wait(mpil_request, MPI_STATUS_IGNORE);
        MPIL_Request_free(&mpil_request);
        compare_results(pmpi_data, mpil_data, s);
#endif
    }

    MPIL_Info_free(&mpil_info);
    MPIL_Comm_free(&mpil_comm);

    MPIL_Finalize();
    MPI_Finalize();
    return 0;
}

