#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>

double allreduce(int size, float* sendbuf_d, float* sendbuf, float* recvbuf_d, float* recvbuf, 
        MPI_Comm comm, int n_iters)
{
    gpuDeviceSynchronize();
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        gpuMemcpy(sendbuf, sendbuf_d, size*sizeof(float), gpuMemcpyDeviceToHost);
        MPI_Allreduce(sendbuf, recvbuf, size, MPI_FLOAT, MPI_SUM, comm);
        gpuMemcpy(recvbuf_d, recvbuf, size*sizeof(float), gpuMemcpyHostToDevice);
    }
    double tfinal = (MPI_Wtime() - t0) / n_iters;
    return tfinal;
}

int estimate_iters(int size, float* sendbuf_d, float* sendbuf, float* recvbuf_d, float* recvbuf, MPI_Comm comm)
{
    // Warm-Up
    allreduce(size, sendbuf_d, sendbuf, recvbuf_d, recvbuf, comm, 1);

    // Time 2 Iterations
    double time = allreduce(size, sendbuf_d, sendbuf, recvbuf_d, recvbuf, comm, 2);
    
    // Get Max Time Across All Procs
    MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);    

    // Set NIters so Timing ~ 1 Second
    int n_iters = (2.0 / time) + 1;
    return n_iters;
}

double time_allreduce(int size, float* sendbuf_d, float* sendbuf, float* recvbuf_d, float* recvbuf, MPI_Comm comm)
{
    int n_iters = estimate_iters(size, sendbuf_d, sendbuf, recvbuf_d, recvbuf, comm);
    double time = allreduce(size, sendbuf_d, sendbuf, recvbuf_d, recvbuf, comm, n_iters);
    return time;
}

void print_allreduce(int max_p, float* sendbuf_d, float* sendbuf, float* recvbuf_d, float* recvbuf, float* recvbuf_std, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (int i = 0; i < max_p; i++)
    { 
        int s = pow(2, i);
        if (rank == 0) printf("Size %d: ", s);

        // Check Correctness
        MPI_Allreduce(sendbuf_d, recvbuf_d, s, MPI_FLOAT, MPI_SUM, comm);
        gpuMemcpy(recvbuf_std, recvbuf_d, s*sizeof(float), gpuMemcpyDeviceToHost);
        allreduce(s, sendbuf_d, sendbuf, recvbuf_d, recvbuf, comm, 1);
        gpuMemcpy(recvbuf, recvbuf_d, s*sizeof(float), gpuMemcpyDeviceToHost);
        for (int i = 0; i < s; i++)
            if (fabs(recvbuf_std[i] - recvbuf[i]) > 1e-06)
            {
                printf("Different RESULTS!\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

        // Time Copy-to-CPU Allreduce
        double time = time_allreduce(s, sendbuf_d, sendbuf, recvbuf_d, recvbuf, comm);
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


    // Set Local GPU
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 
            rank, MPI_INFO_NULL, &local_comm);
    int local_rank, ppn;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &ppn);

    int gpn;
    gpuGetDeviceCount(&gpn);

    int ppg = ppn / gpn;
    int local_gpu = local_rank / ppg;
    int gpu_rank = local_rank % ppg;

    int max_p = 30;
    int max_s = pow(2, max_p);
    int max_s_proc = max_s / ppg;

    gpuSetDevice(local_gpu);
    
    float* sendbuf_d;
    float* recvbuf_d;
    float* sendbuf;
    float* recvbuf;
    float* recvbuf_std;

    gpuMalloc((void**)&sendbuf_d, max_s*sizeof(float));
    gpuMalloc((void**)&recvbuf_d, max_s*sizeof(float));
    gpuMallocHost((void**)&sendbuf, max_s_proc*sizeof(float));
    gpuMallocHost((void**)&recvbuf, max_s_proc*sizeof(float));
    gpuMallocHost((void**)&recvbuf_std, max_s_proc*sizeof(float));

    if (rank == 0) printf("Starting Allreduce Timings:\n");
    print_allreduce(max_p, sendbuf_d, sendbuf, recvbuf_d, recvbuf, recvbuf_std, MPI_COMM_WORLD);

    gpuFree(sendbuf_d);
    gpuFree(recvbuf_d);
    gpuFreeHost(sendbuf);
    gpuFreeHost(recvbuf);
    gpuFreeHost(recvbuf_std);

    MPI_Comm_free(&local_comm);

    MPI_Finalize();
    return 0;
}
