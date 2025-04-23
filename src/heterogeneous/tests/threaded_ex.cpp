#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>

void threaded_sum(void*, void*, int*, MPI_Datatype);

// Currently hardcoded for floats
void threaded_sum(void* in, void* out, int* len, MPI_Datatype type)
{
    float* input = (float*)in;
    float* output = (float*)out;

#pragma omp parallel for
    for (int i = 0; i < *len; i++)
        output[i] += input[i];
}

double allreduce(int size, float* sendbuf_d, float* sendbuf, float* recvbuf_d, float* recvbuf, 
        MPI_Comm comm, int n_iters, MPI_Op operation)
{
    gpuDeviceSynchronize();
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        gpuMemcpy(sendbuf, sendbuf_d, size*sizeof(float), gpuMemcpyDeviceToHost);
        MPI_Allreduce(sendbuf, recvbuf, size, MPI_FLOAT, operation, comm);
        gpuMemcpy(recvbuf_d, recvbuf, size*sizeof(float), gpuMemcpyHostToDevice);
    }
    double tfinal = (MPI_Wtime() - t0) / n_iters;
    return tfinal;
}

void print_allreduce(int max_p, float* sendbuf_d, float* sendbuf, float* recvbuf_d, float* recvbuf, float* recvbuf_std,
        MPI_Comm comm, int ppg, int socket_rank)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   
    double max_time;

    MPI_Op operation;
    MPI_Op_create((MPI_User_function *)&threaded_sum, 1, &operation);

    for (int i = 4; i < max_p; i++)
    { 
        double time = 0;
        int s_g = pow(2, i);
        int s = s_g / ppg;
        if (rank == 0) printf("Size %d: ", s_g);

        // Compare Results
        // 1. Multi-Proc
        if (socket_rank < ppg)
            allreduce(s, &(sendbuf_d[socket_rank*s]), sendbuf, &(recvbuf_d[socket_rank*s]), recvbuf, comm, 1, operation);
        // 2. Standard
        if (socket_rank == 0) 
        {
            cudaMemcpy(recvbuf, recvbuf_d, s*sizeof(float), cudaMemcpyDeviceToHost);
            allreduce(s, sendbuf_d, sendbuf, recvbuf_d, recvbuf, comm, 1, MPI_SUM);
            cudaMemcpy(recvbuf_std, recvbuf_d, s*sizeof(float), cudaMemcpyDeviceToHost);
            for (int i = 0; i < s; i++)
                if (fabs(recvbuf[i] - recvbuf_std[i]) > 1e-6) printf("DIFFERENCE IN RESULTS!\n");
        }
     
            
        // Warm-Up
        if (socket_rank < ppg)
            allreduce(s, &(sendbuf_d[socket_rank*s]), sendbuf, &(recvbuf_d[socket_rank*s]), recvbuf, comm, 1, operation);

        // Time 2 iterations
        if (socket_rank < ppg)
            time = allreduce(s, &(sendbuf_d[socket_rank*s]), sendbuf, &(recvbuf_d[socket_rank*s]), recvbuf, comm, 2, operation);

        MPI_Barrier(MPI_COMM_WORLD);
        
        // Get Max Time
        MPI_Allreduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);    

        // Set NIters so Timing ~ 1 Second
        int n_iters = (2.0 / max_time) + 1;

        // Time Allreduce
        if (socket_rank < ppg)
            time = allreduce(s, &(sendbuf_d[socket_rank*s]), sendbuf, &(recvbuf_d[socket_rank*s]), recvbuf, comm, n_iters, operation);

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Allreduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);    
        if (rank == 0) printf("%e\n", max_time);
    }
    if (rank == 0) printf("\n");

    MPI_Op_free(&operation);
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
    int local_gpu = rank / ppg;
    int gpu_rank = rank % ppg;

    int max_p = 30;
    int max_s = pow(2, max_p);
    int max_s_proc = max_s / ppg;

    gpuSetDevice(local_gpu);
    
    MPI_Comm gpu_comm;
    MPI_Comm_split(MPI_COMM_WORLD, gpu_rank, rank, &gpu_comm);

    MPI_Comm socket_comm;
    MPI_Comm_split(local_comm, local_gpu, rank, &socket_comm);

    float* sendbuf_d;
    float* recvbuf_d;
    float* sendbuf;
    float* recvbuf;
    float* recvbuf_std;


    gpuMallocHost((void**)&sendbuf, max_s*sizeof(float));
    gpuMallocHost((void**)&recvbuf, max_s_proc*sizeof(float));
    gpuMallocHost((void**)&recvbuf_std, max_s_proc*sizeof(float));

    cudaIpcMemHandle_t send_handle, recv_handle;
    if (gpu_rank == 0)
    {
        gpuMalloc((void**)&sendbuf_d, max_s*sizeof(float));
        gpuMalloc((void**)&recvbuf_d, max_s*sizeof(float));

        for (int i = 0; i < max_s; i++)
            sendbuf[i] = ((float)rand()) / RAND_MAX;
        gpuMemcpy(sendbuf_d, sendbuf, max_s*sizeof(float), gpuMemcpyHostToDevice);

        cudaIpcGetMemHandle(&send_handle, sendbuf_d);
        cudaIpcGetMemHandle(&recv_handle, recvbuf_d);
    }
    MPI_Bcast(&send_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, socket_comm);
    MPI_Bcast(&recv_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, socket_comm);
    if (gpu_rank != 0)
    {
        cudaIpcOpenMemHandle((void**)&sendbuf_d, send_handle, cudaIpcMemLazyEnablePeerAccess); 
        cudaIpcOpenMemHandle((void**)&recvbuf_d, recv_handle, cudaIpcMemLazyEnablePeerAccess); 
    }

    for (int i = 0; i < 5; i++)
    { 
i = 4;
        int ppg = pow(2, i);
        if (rank == 0) printf("Starting Allreduce Timings, PPG %d:\n", ppg);
        print_allreduce(max_p, sendbuf_d, sendbuf, recvbuf_d, recvbuf, recvbuf_std, gpu_comm, ppg, gpu_rank);
        MPI_Barrier(MPI_COMM_WORLD);
break;
    }

    if (gpu_rank == 0)
    {
        gpuFree(sendbuf_d);
        gpuFree(recvbuf_d);
    }
    else
    {
        cudaIpcCloseMemHandle(sendbuf_d);
        cudaIpcCloseMemHandle(recvbuf_d);
    }
    gpuFreeHost(sendbuf);
    gpuFreeHost(recvbuf);

    MPI_Comm_free(&socket_comm);
    MPI_Comm_free(&gpu_comm);
    MPI_Comm_free(&local_comm);

    MPI_Finalize();
    return 0;
}
