#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>

double allreduce_cpu(int size, float* sendbuf_d, float* sendbuf,
        float* recvbuf_d, float* recvbuf, MPI_Comm comm)
{
    gpuMemcpy(sendbuf, sendbuf_d, size*sizeof(float), gpuMemcpyDeviceToHost);
    MPI_Allreduce(sendbuf, recvbuf, size, MPI_FLOAT, MPI_SUM, comm);
    gpuMemcpy(recvbuf_d, recvbuf, size*sizeof(float), gpuMemcpyHostToDevice);
}

double allreduce_gpu(int size, float* sendbuf_d, float* recvbuf_d, 
        MPI_Comm comm)
{
    MPI_Allreduce(sendbuf_d, recvbuf_d, size, MPI_FLOAT, MPI_SUM, comm);
}

double allreduce(int size, int gpu,
        float* sendbuf_d, float* sendbuf, 
        float* recvbuf_d, float* recvbuf, 
        MPI_Comm comm, int n_iters)
{
    MPI_Request gpu_req;

    gpuDeviceSynchronize();
    MPI_Barrier(comm);
    double t0, tfinal;

    if (gpu)
    {
        t0 = MPI_Wtime();
        for (int i = 0; i < n_iters; i++)
        {
            allreduce_gpu(size, sendbuf_d, recvbuf_d, comm);
        }
        tfinal = (MPI_Wtime() - t0) / n_iters;
    }
    else
    {  
        t0 = MPI_Wtime();
        for (int i = 0; i < n_iters; i++)
        {
            allreduce_cpu(size, sendbuf_d, sendbuf, recvbuf_d, recvbuf, comm);
        }
        tfinal = (MPI_Wtime() - t0) / n_iters;
    }
    return tfinal;
}

void print_allreduce(int max_p, float* sendbuf_d, float* sendbuf, 
        float* recvbuf_d, float* recvbuf, float* recvbuf_std,
        MPI_Comm comm, int ppg, int socket_rank)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   
    double max_time;

    int start_i = 3+log2(ppg);
    int gpu = 0;

    for (int i = start_i; i < max_p; i++)
    { 
        double time = 0;
        int s_g = pow(2, i);
        int s = s_g / ppg;
        if (1.0*socket_rank / ppg < 3.0/4)
            gpu = 1;

        if (rank == 0) printf("Size %d: ", s_g);

        int first = s * socket_rank;

        // Compare Results
        // 1. Multi-Proc
        allreduce(s, gpu, &(sendbuf_d[first]), sendbuf, &(recvbuf_d[first]), recvbuf, comm, 1);
        if (socket_rank == 0) 
        {
            MPI_Allreduce(sendbuf_d, recvbuf_d, s_g, MPI_FLOAT, MPI_SUM, comm);
        }
        // Copy equal portions for comparisons
        int compare_s = s_g / ppg;
        gpuMemcpy(recvbuf_std, &(recvbuf_d[socket_rank*compare_s]), compare_s*sizeof(float), gpuMemcpyDeviceToHost);
        gpuMemcpy(recvbuf, &(recvbuf_d[socket_rank*compare_s]), compare_s*sizeof(float), gpuMemcpyDeviceToHost);
        for (int i = 0; i < compare_s; i++)
            if (fabs(recvbuf[i] - recvbuf_std[i]) > 1e-6) 
            {
                printf("DIFFERENCE IN RESULTS! %e vs %e\n", recvbuf[i], recvbuf_std[i]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
     
        // Warm-Up
        allreduce(s, gpu, &(sendbuf_d[first]), sendbuf, &(recvbuf_d[first]), recvbuf, comm, 1);

        // Time 2 iterations
        time = allreduce(s, gpu, &(sendbuf_d[first]), sendbuf, &(recvbuf_d[first]), recvbuf, comm, 2);

        // Get Max Time
        MPI_Allreduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);    

        // Set NIters so Timing ~ 1 Second
        int n_iters = (2.0 / max_time) + 1;

        // Time Allreduce
        time = allreduce(s, gpu, &(sendbuf_d[first]), sendbuf, &(recvbuf_d[first]), recvbuf, comm, n_iters);

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
    gpuGetDeviceCount(&gpn);

    int ppg = ppn / gpn;
    int local_gpu = local_rank / ppg;
    int gpu_rank = local_rank % ppg;

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


    gpuMallocHost((void**)&sendbuf, max_s_proc*sizeof(float));
    gpuMallocHost((void**)&recvbuf, max_s_proc*sizeof(float));
    gpuMallocHost((void**)&recvbuf_std, max_s_proc*sizeof(float));

    cudaIpcMemHandle_t send_handle, recv_handle;
    if (gpu_rank == 0)
    {
        gpuMalloc((void**)&sendbuf_d, max_s*sizeof(float));
        gpuMalloc((void**)&recvbuf_d, max_s*sizeof(float));

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

    for (int i = 0; i < max_s_proc; i++)
        sendbuf[i] = ((float)rand()) / RAND_MAX;
    gpuMemcpy(&(sendbuf_d[gpu_rank*max_s]), sendbuf, max_s*sizeof(float), gpuMemcpyHostToDevice);

    if (rank == 0) printf("Starting Allreduce Timings, PPG %d:\n", ppg);
    print_allreduce(max_p, sendbuf_d, sendbuf, recvbuf_d, recvbuf, recvbuf_std, gpu_comm, ppg, gpu_rank);
    MPI_Barrier(MPI_COMM_WORLD);

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
