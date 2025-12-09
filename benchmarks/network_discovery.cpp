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
    printf("Rank: %d\n", rank);

    int tag;
    MPIX_Comm_tag(xcomm, &tag);
    printf("Tag: %d\n", tag);

    printf("Starting network discover\n");
    int max_p = 11;
    for (int k = 0; k < max_p; k++)
    {
        int size = 1 << k;
        printf("Testing with message size: %d\n", k);
        double* adjacencyMatrix = network_discovery(xcomm, size, tag, 100);
        printf("Adjacency matrix (message size: %d)\n", size);
        for (int i = 0; i < num_procs; i++)
        {
            printf("%.10lf\t", adjacencyMatrix[i]);            
        }

        printf("\n");
        free(adjacencyMatrix);
    }
}
