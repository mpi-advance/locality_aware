#include <mpi.h>

#include "mpi_advance.h"

double* network_discovery(char* send_buffer, char* recv_buffer, double* times,
        int size, int tag, int num_iterations)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int proc;
    MPI_Status status;

    for (int i = 0; i < num_procs - 1; i++)
    {
        int dist = i / 2 + 1;
        // warm up
        printf("Warming up\n");
        if ((rank / dist) % 2 == i % 2)
        {
            proc = (rank + dist) % num_procs;
            MPI_Send(send_buffer, size, MPI_CHAR, send_proc, tag, MPI_COMM_WORLD);
            MPI_Recv(recv_buffer, size, MPI_CHAR, send_proc, tag, MPI_COMM_WORLD, &status);
        }
        else
        {
            proc = (rank - dist + num_procs) % num_procs;
            MPI_Recv(recv_buffer, size, MPI_CHAR, send_proc, tag, MPI_COMM_WORLD, &status);
            MPI_Send(send_buffer, size, MPI_CHAR, send_proc, tag, MPI_COMM_WORLD);   
        }

        printf("Testing distance from %d to %d\n", rank, send_proc);
        double t0 = MPI_Wtime();
        for (int j = 0; j < num_iterations; j++)
        {
            if ((rank / dist) % 2 == i % 2)
            {
                MPI_Send(send_buffer, size, MPI_CHAR, send_proc, tag, MPI_COMM_WORLD);
                MPI_Recv(recv_buffer, size, MPI_CHAR, send_proc, tag, MPI_COMM_WORLD, &status);
            }
            else
            {
                MPI_Recv(recv_buffer, size, MPI_CHAR, send_proc, tag, MPI_COMM_WORLD, &status);
                MPI_Send(send_buffer, size, MPI_CHAR, send_proc, tag, MPI_COMM_WORLD);   
            }
        }
        times[i] = MPI_Wtime() - t0 / (2. * num_iterations);
    }

    return times;
}

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
    int max_size = 1 << (max_p-1);
    char* send_buffer = (char*) malloc(num_procs * max_size * sizeof(char));
    char* recv_buffer = (char*) malloc(num_procs * max_size * sizeof(char));
    double* times = (double*)malloc(num_procs * sizeof(double));
    for (int k = 0; k < max_p; k++)
    {
        int size = 1 << k;
        printf("Testing with message size: %d\n", k);
        double* adjacencyMatrix = network_discovery(send_buffer, recv_buffer, times, size, tag, 100);
        printf("Adjacency matrix (message size: %d)\n", size);
        for (int i = 0; i < num_procs; i++)
        {
            printf("%.10lf\t", adjacencyMatrix[i]);            
        }

        printf("\n");
        free(adjacencyMatrix);
    }
    free(send_buffer);
    free(recv_buffer);
    free(times);
}
