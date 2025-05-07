#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>

void cudaCheck(cudaError_t ierr)
{
    if (ierr != cudaSuccess)
    {
        printf("ERROR! %s\n", cudaGetErrorString(ierr));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

double allreduce(int size, float* sendbuf, float* sendbuf_loc, 
        float* recvbuf, float* recvbuf_loc, MPI_Comm comm, int n_iters)
{
    MPI_Request gpu_req;
    gpuDeviceSynchronize();
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        cudaMemcpy(sendbuf_loc, sendbuf, size*sizeof(float), cudaMemcpyDeviceToDevice);
        MPI_Allreduce(sendbuf_loc, recvbuf_loc, size, MPI_FLOAT, MPI_SUM, comm);
        cudaMemcpy(recvbuf, recvbuf_loc, size*sizeof(float), cudaMemcpyDeviceToDevice);
    }
    double tfinal = (MPI_Wtime() - t0) / n_iters;
    return tfinal;
}

void print_allreduce(int max_p, float* sendbuf_d, float* sendbuf_d_local, 
        float* recvbuf_d, float* recvbuf_d_local, float* recvbuf, float* recvbuf_std,
        MPI_Comm comm, int ppg, int socket_rank)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   
    double max_time;

    int start_i = log2(ppg);

    for (int i = start_i; i < max_p; i++)
    { 
        double time = 0;
        int s_g = pow(2, i);
        int s = s_g / ppg;
        if (rank == 0) printf("Size %d (Per Proc %d): ", s_g, s);

        // Compare Results
        // 1. Multi-Proc
        allreduce(s, &(sendbuf_d[s*socket_rank]), sendbuf_d_local,
                &(recvbuf_d[s*socket_rank]), recvbuf_d_local,  comm, 1);
        gpuMemcpy(recvbuf_std, &(recvbuf_d[socket_rank*s]), s*sizeof(float), gpuMemcpyDeviceToHost);

        if (socket_rank == 0) 
        {
            MPI_Allreduce(sendbuf_d, recvbuf_d, s_g, MPI_FLOAT, MPI_SUM, comm);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        gpuMemcpy(recvbuf, &(recvbuf_d[socket_rank*s]), s*sizeof(float), gpuMemcpyDeviceToHost);
        for (int i = 0; i < s; i++)
            if (fabs(recvbuf[i] - recvbuf_std[i]) > 1e-6) 
            {
                printf("DIFFERENCE IN RESULTS! %e vs %e\n", recvbuf[i], recvbuf_std[i]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
     
        // Warm-Up
        allreduce(s, &(sendbuf_d[s*socket_rank]), sendbuf_d_local,
                &(recvbuf_d[s*socket_rank]), recvbuf_d_local,  comm, 1);

        // Time 2 iterations
        time = allreduce(s, &(sendbuf_d[s*socket_rank]), sendbuf_d_local,
                &(recvbuf_d[s*socket_rank]), recvbuf_d_local,  comm, 2);

        // Get Max Time
        MPI_Allreduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);    

        // Set NIters so Timing ~ 1 Second
        int n_iters = (2.0 / max_time) + 1;

        // Time Allreduce
        time = allreduce(s, &(sendbuf_d[s*socket_rank]), sendbuf_d_local,
                &(recvbuf_d[s*socket_rank]), recvbuf_d_local,  comm, n_iters);

        MPI_Allreduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);    
        if (rank == 0) printf("%e\n", max_time);
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
    cudaCheck(cudaGetDeviceCount(&gpn));

    int ppg = ppn / gpn;
    int local_gpu = local_rank / ppg;
    int gpu_rank = local_rank % ppg;

    int max_p = 30;
    int max_s = pow(2, max_p);
    int max_s_proc = max_s / ppg;

    cudaCheck(cudaSetDevice(local_gpu));
    
    MPI_Comm gpu_comm;
    MPI_Comm_split(MPI_COMM_WORLD, gpu_rank, rank, &gpu_comm);

    MPI_Comm socket_comm;
    MPI_Comm_split(local_comm, local_gpu, rank, &socket_comm);

    float* sendbuf_d;
    float* recvbuf_d;
    float* sendbuf;
    float* recvbuf;
    float* recvbuf_std;
    float* sendbuf_d_local;
    float* recvbuf_d_local;

    cudaCheck(cudaMallocHost((void**)&sendbuf, max_s_proc*sizeof(float)));
    cudaCheck(cudaMallocHost((void**)&recvbuf, max_s_proc*sizeof(float)));
    cudaCheck(cudaMallocHost((void**)&recvbuf_std, max_s_proc*sizeof(float)));

    cudaCheck(cudaMalloc((void**)&sendbuf_d_local, max_s_proc*sizeof(float)));
    cudaCheck(cudaMalloc((void**)&recvbuf_d_local, max_s_proc*sizeof(float)));

    cudaIpcMemHandle_t send_handle, recv_handle;
    if (gpu_rank == 0)
    {
        cudaCheck(cudaMalloc((void**)&sendbuf_d, max_s*sizeof(float)));
        cudaCheck(cudaMalloc((void**)&recvbuf_d, max_s*sizeof(float)));

        cudaCheck(cudaIpcGetMemHandle(&send_handle, sendbuf_d));
        cudaCheck(cudaIpcGetMemHandle(&recv_handle, recvbuf_d));
    }
    MPI_Bcast(&send_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, socket_comm);
    MPI_Bcast(&recv_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, socket_comm);
    if (gpu_rank != 0)
    {
        cudaCheck(cudaIpcOpenMemHandle((void**)&sendbuf_d, send_handle, cudaIpcMemLazyEnablePeerAccess)); 
        cudaCheck(cudaIpcOpenMemHandle((void**)&recvbuf_d, recv_handle, cudaIpcMemLazyEnablePeerAccess)); 
    }

    for (int i = 0; i < max_s_proc; i++)
        sendbuf[i] = ((float)rand()) / RAND_MAX;
    cudaCheck(cudaMemcpy(&(sendbuf_d[gpu_rank*max_s_proc]), sendbuf, max_s_proc*sizeof(float), cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);
    cudaCheck(cudaDeviceSynchronize());

    if (rank == 0) printf("Starting Allreduce Timings, PPG %d:\n", ppg);
    print_allreduce(max_p, sendbuf_d, sendbuf_d_local, recvbuf_d, recvbuf_d_local, 
            recvbuf, recvbuf_std, gpu_comm, ppg, gpu_rank);
    MPI_Barrier(MPI_COMM_WORLD);

    if (gpu_rank == 0)
    {
        cudaCheck(cudaFree(sendbuf_d));
        cudaCheck(cudaFree(recvbuf_d));
    }
    else
    {
        cudaCheck(cudaIpcCloseMemHandle(sendbuf_d));
        cudaCheck(cudaIpcCloseMemHandle(recvbuf_d));
    }

    cudaCheck(cudaFree(sendbuf_d_local));
    cudaCheck(cudaFree(recvbuf_d_local));

    cudaCheck(cudaFreeHost(sendbuf));
    cudaCheck(cudaFreeHost(recvbuf));
    cudaCheck(cudaFreeHost(recvbuf_std));

    MPI_Comm_free(&socket_comm);
    MPI_Comm_free(&gpu_comm);
    MPI_Comm_free(&local_comm);

    MPI_Finalize();
    return 0;
}
