#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>

void compare_allgather_results(int* pmpi, int* mpix, int s, int rank)
{
    for (int i = 0; i < s; i++)
    {
        if (pmpi[i] != mpix[i])
        {
            fprintf(stderr, "Rank: %d, size: %d, index: %d MPIX Allgather != pmpi, pmpi: %d, mpix: %d\n", rank, s, i, pmpi[i], mpix[i]);
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

    // Test integer Allgather
    int max_i = 10;
    int max_s = pow(2, max_i); 
    srand(time(NULL));
    std::vector<int> local_data(max_s * num_procs);

    MPIX_Comm *locality_comm;
    MPIX_Comm_init(&locality_comm, MPI_COMM_WORLD);
    MPIX_Comm_topo_init(locality_comm);
    MPIX_Comm_leader_init(locality_comm, 4);
    for (int i = 0; i < 6; i++)
    {
        // int i = 0;
        int s = pow(2, i);
        if (rank == 0)
            printf("*************\nSize: %d\n*************\n", s);
        
        for (int j = 0; j < s; j++)
        {
            local_data[j] = (j + 1) * (rank + 1) - 1;
        }

        // standard Allgather
        int *pmpi_allgather = (int*) malloc(s * num_procs * sizeof(int));
        PMPI_Allgather(local_data.data(), s, MPI_INT, pmpi_allgather, s, MPI_INT, MPI_COMM_WORLD);

        int* mpix_allgather = (int*) malloc(s * num_procs * sizeof(int));
        if (rank == 0)
            printf("allgather multileader\n");
        allgather_multileader(local_data.data(), s, MPI_INT, mpix_allgather, s, MPI_INT, *locality_comm);
        compare_allgather_results(pmpi_allgather, mpix_allgather, s * num_procs, rank);

        if (rank == 0)
            printf("allgather hierarchical\n");
        allgather_hierarchical(local_data.data(), s, MPI_INT, mpix_allgather, s, MPI_INT, *locality_comm);
        compare_allgather_results(pmpi_allgather, mpix_allgather, s * num_procs, rank);

        if (rank == 0)
            printf("allgather locality aware\n");
        allgather_locality_aware(local_data.data(), s, MPI_INT, mpix_allgather, s, MPI_INT, *locality_comm);
        compare_allgather_results(pmpi_allgather, mpix_allgather, s * num_procs, rank);

        if (rank == 0)
            printf("allgather node aware\n");
        allgather_node_aware(local_data.data(), s, MPI_INT, mpix_allgather, s, MPI_INT, *locality_comm);
        compare_allgather_results(pmpi_allgather, mpix_allgather, s * num_procs, rank);

        if (rank == 0)
            printf("allgather multileader locality aware\n");
        allgather_multileader_locality_aware(local_data.data(), s, MPI_INT, mpix_allgather, s, MPI_INT, *locality_comm);
        compare_allgather_results(pmpi_allgather, mpix_allgather, s * num_procs, rank);

        free(pmpi_allgather);
        free(mpix_allgather);
    }
}