#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>

double allreduce(int size, float* sendbuf, float* recvbuf, MPI_Comm comm, int n_iters)
{
    MPI_Request gpu_req;
    gpuDeviceSynchronize();
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        MPI_Allreduce(sendbuf, recvbuf, size, MPI_FLOAT, MPI_SUM, comm);
    }
    double tfinal = (MPI_Wtime() - t0) / n_iters;
    return tfinal;
}

int estimate_iters(int size, float* sendbuf, float* recvbuf, MPI_Comm comm)
{
    // Warm-Up
    allreduce(size, sendbuf, recvbuf, comm, 1);

    // Time 2 Iterations
    double time = allreduce(size, sendbuf, recvbuf, comm, 2);
    
    // Get Max Time Across All Procs
    MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, comm);    

    // Set NIters so Timing ~ 1 Second
    int n_iters = (2.0 / time) + 1;
    return n_iters;
}

double time_allreduce(int size, float* sendbuf, float* recvbuf, MPI_Comm comm)
{
    int n_iters = estimate_iters(size, sendbuf, recvbuf, comm);
    double time = allreduce(size, sendbuf, recvbuf, comm, n_iters);
    return time;
}

void print_allreduce(int max_p, float* sendbuf, float* recvbuf, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    for (int i = 0; i < max_p; i++)
    { 
        int s = pow(2, i);
        if (rank == 0) printf("Size %d: ", s);
        double time = time_allreduce(s, sendbuf, recvbuf, comm);
        MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, comm);
        if (rank == 0) printf("%e\n", time);
    }
    if (rank == 0) printf("\n");
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int max_p = 30;
    int max_s = pow(2, max_p);

    // Set Local GPU
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 
            rank, MPI_INFO_NULL, &local_comm);
    int local_rank, ppn;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &ppn);
    MPI_Comm_free(&local_comm);

    int gpn;
    gpuGetDeviceCount(&gpn);

    int ppg = ppn / gpn;
    int local_gpu = local_rank / ppg;
    int gpu_rank = local_rank % ppg;

    gpuSetDevice(local_gpu);

    float* sendbuf;
    float* recvbuf;

    gpuMalloc((void**)&sendbuf, max_s*sizeof(float));
    gpuMalloc((void**)&recvbuf, max_s*sizeof(float));

    print_allreduce(max_p, sendbuf, recvbuf, MPI_COMM_WORLD);

    gpuFree(sendbuf);
    gpuFree(recvbuf);

    MPI_Finalize();
    return 0;
}
