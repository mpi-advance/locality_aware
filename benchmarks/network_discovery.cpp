#include <mpi.h>

#include "mpi_advance.h"

double* network_discovery(char* send_buffer, char* recv_buffer, int size, int tag, int num_iterations)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int proc;
    MPI_Status status;

    double* times = (double*) malloc(num_procs * sizeof(double));
    times[0] = 0.0;
    for (int i = 0; i < num_procs - 1; i++)
    {
        int dist = i / 2 + 1;
        // warm up
        // printf("Warming up\n");
        if ((rank / dist) % 2 == i % 2)
        {
            proc = (rank + dist) % num_procs;
            MPI_Send(send_buffer, size, MPI_CHAR, proc, tag, MPI_COMM_WORLD);
            MPI_Recv(recv_buffer, size, MPI_CHAR, proc, tag, MPI_COMM_WORLD, &status);
        }
        else
        {
            proc = (rank - dist + num_procs) % num_procs;
            MPI_Recv(recv_buffer, size, MPI_CHAR, proc, tag, MPI_COMM_WORLD, &status);
            MPI_Send(send_buffer, size, MPI_CHAR, proc, tag, MPI_COMM_WORLD);   
        }

        // if (rank == 0)
        // {
        //     printf("Sending to %d\n", proc);
        // }

        // printf("Testing distance from %d to %d\n", rank, proc);
        double t0 = MPI_Wtime();
        for (int j = 0; j < num_iterations; j++)
        {
            if ((rank / dist) % 2 == i % 2)
            {
                MPI_Send(send_buffer, size, MPI_CHAR, proc, tag, MPI_COMM_WORLD);
                MPI_Recv(recv_buffer, size, MPI_CHAR, proc, tag, MPI_COMM_WORLD, &status);
            }
            else
            {
                MPI_Recv(recv_buffer, size, MPI_CHAR, proc, tag, MPI_COMM_WORLD, &status);
                MPI_Send(send_buffer, size, MPI_CHAR, proc, tag, MPI_COMM_WORLD);   
            }
        }
        times[proc] = MPI_Wtime() - t0 / (2. * num_iterations);
    }

    return times;
}

double* pingPong(char* send_buffer, char* recv_buffer, int size, int tag, int num_iterations)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int proc;
    MPI_Status status;

    double* times = (double*) malloc(num_procs * sizeof(double));
    times[0] = 0.0;
    for (int i = 1; i < num_procs - 1; i++)
    {
        // warm up
        if (rank == 0)
        {
            MPI_Send(send_buffer, size, MPI_CHAR, i, tag, MPI_COMM_WORLD);
            MPI_Recv(recv_buffer, size, MPI_CHAR, i, tag, MPI_COMM_WORLD, &status);
        }
        else if (rank == i)
        {
            MPI_Recv(recv_buffer, size, MPI_CHAR, 0, tag, MPI_COMM_WORLD, &status);
            MPI_Send(send_buffer, size, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
        }

        // test
        double t0 = MPI_Wtime();
        for (int j = 0; j < num_iterations; j++)
        {
            if (rank == 0)
            {
                MPI_Send(send_buffer, size, MPI_CHAR, i, tag, MPI_COMM_WORLD);
                MPI_Recv(recv_buffer, size, MPI_CHAR, i, tag, MPI_COMM_WORLD, &status);
            }
            else if (rank == i)
            {
                MPI_Recv(recv_buffer, size, MPI_CHAR, 0, tag, MPI_COMM_WORLD, &status);
                MPI_Send(send_buffer, size, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
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

    int tag;
    MPIX_Comm_tag(xcomm, &tag);

    int max_p = 11;
    int max_size = 1 << (max_p-1);
    char* send_buffer = (char*) malloc(num_procs * max_size * sizeof(char));
    char* recv_buffer = (char*) malloc(num_procs * max_size * sizeof(char));
    // double* times = (double*)malloc(num_procs * sizeof(double));
    for (int k = 0; k < max_p; k++)
    {
        int size = 1 << k;
        double* times = pingPong(send_buffer, recv_buffer, size, tag, 100);
        // double* times = network_discovery(send_buffer, recv_buffer, size, tag, 100);
        if (rank == 0)
        {
            printf("Adjacency matrix (message size: %d)\n", size);
            for (int i = 0; i < num_procs; i++)
            {
                printf("%.10lf\t", times[i]);            
            }

            printf("\n");
        }

        free(times);
    }
    free(send_buffer);
    free(recv_buffer);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);

    int node_rank, ppn;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &ppn);

    MPI_Comm group_comm;
    MPI_Comm_split(MPI_COMM_WORLD, node_rank, rank, &group_comm);

    int node;
    MPI_Comm_rank(group_comm, &node);
    printf("Rank %d has local rank %d of %d on node %d\n", rank, node_rank, ppn, node);
}
