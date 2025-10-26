#include <mpi.h>

#include "mpi_advance.h"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    MPIX_Comm *xcomm;
    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int tag;
    MPIX_Comm_tag(xcomm, &tag);

    int max_p = 11;
    for (int k = 0; k < max_p; k++)
    {
        int size = 1 << k;
        double* adjacencyMatrix = network_discovery(xcomm, size, tag, 100);
        if (rank == 0)
        {
            printf("Adjacency matrix (message size: %d)\n", size);
            for (int i = 0; i < num_procs; i++)
            {
                for (int j = 0; j < num_procs; j++)
                {
                    printf("%.6lf\t", adjacencyMatrix[i * num_procs + j]);            
                }
                printf("\n");
            }
        }    
        free(adjacencyMatrix);
    }
}