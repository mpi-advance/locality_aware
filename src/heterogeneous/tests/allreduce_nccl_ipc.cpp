#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>
#include "nccl.h"

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

double allreduce(int size, float* sendbuf, float* recvbuf, ncclComm_t nccl_comm, 
        cudaStream_t stream, MPI_Comm comm, int n_iters)
{
    cudaStreamSynchronize(stream);
    gpuDeviceSynchronize();
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        NCCLCHECK(ncclAllReduce((const void*) sendbuf, (void*)recvbuf, size, ncclFloat,
                ncclSum, nccl_comm, stream));
        CUDACHECK(cudaStreamSynchronize(stream));
    }
    double tfinal = (MPI_Wtime() - t0) / n_iters;
    return tfinal;
}

void print_allreduce(int max_p, float* sendbuf, float* recvbuf, 
        float* recvbuf_std, float* recvbuf_ipc,
        MPI_Comm comm, ncclComm_t nccl_comm, cudaStream_t stream, 
        int gpu_rank, int ppg)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int start_i = log2(ppg);
    for (int i = start_i; i < max_p; i++)
    { 
        int s_g = pow(2, i);
        int s = s_g / ppg;
        if (rank == 0) printf("Size %d: ", s_g);
        CUDACHECK(cudaDeviceSynchronize());

        if (gpu_rank == 0)
        {
            NCCLCHECK(ncclAllReduce(sendbuf, recvbuf, s_g, ncclFloat, ncclSum, nccl_comm, stream));
            CUDACHECK(cudaStreamSynchronize(stream));
        }
        CUDACHECK(cudaDeviceSynchronize());
        MPI_Barrier(comm);
        cudaMemcpy(recvbuf_std, &(recvbuf[s*gpu_rank]), s*sizeof(float), cudaMemcpyDeviceToHost);

        // Warm-Up
        allreduce(s, &(sendbuf[s*gpu_rank]), &(recvbuf[s*gpu_rank]), nccl_comm, stream, comm, 1);
        cudaMemcpy(recvbuf_ipc, &(recvbuf[s*gpu_rank]), s*sizeof(float), cudaMemcpyDeviceToHost);
        CUDACHECK(cudaDeviceSynchronize());
        MPI_Barrier(comm);

        for (int i = 0; i < s; i++)
            if (fabs(recvbuf_ipc[i] - recvbuf_std[i]) > 1e-6)
            {
                printf("DIFFERENCE IN RESULTS! %e vs %e\n", recvbuf_ipc[i], recvbuf_std[i]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

        // Estimate Iterations
        double time = allreduce(s, &(sendbuf[s*gpu_rank]), &(recvbuf[s*gpu_rank]), nccl_comm, stream, comm, 2);
        CUDACHECK(cudaDeviceSynchronize());
        MPI_Barrier(comm);
        MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, comm);
        int n_iters = (2.0/time) + 1;

        // Time Allreduce
        time = allreduce(s, &(sendbuf[s*gpu_rank]), &(recvbuf[s*gpu_rank]), nccl_comm, stream, comm, n_iters);
        CUDACHECK(cudaDeviceSynchronize());
        MPI_Barrier(comm);
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

    int max_p = 31;
    int max_s = pow(2, max_p);

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
    gpuSetDevice(local_gpu);

    int max_s_proc = max_s / ppg;

    MPI_Comm gpu_comm;
    MPI_Comm_split(MPI_COMM_WORLD, gpu_rank, rank, &gpu_comm);

    MPI_Comm socket_comm;
    MPI_Comm_split(local_comm, local_gpu, rank, &socket_comm);

    float *sendbuf, *recvbuf_std, *recvbuf_ipc;
    cudaMallocHost((void**)&sendbuf, max_s_proc*sizeof(float));
    cudaMallocHost((void**)&recvbuf_std, max_s_proc*sizeof(float));
    cudaMallocHost((void**)&recvbuf_ipc, max_s_proc*sizeof(float));
    
    float* sendbuf_d;
    float* recvbuf_d;

    // Allocate sendbuf/recvbuf on rank 0 of gpu_comm
    cudaIpcMemHandle_t send_handle, recv_handle;
    if (gpu_rank == 0)
    {
        CUDACHECK(cudaMalloc((void**)&sendbuf_d, max_s*sizeof(float)));
        CUDACHECK(cudaMalloc((void**)&recvbuf_d, max_s*sizeof(float)));

        CUDACHECK(cudaIpcGetMemHandle(&send_handle, sendbuf_d));
        CUDACHECK(cudaIpcGetMemHandle(&recv_handle, recvbuf_d));
    }
    MPI_Bcast(&send_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, socket_comm);
    MPI_Bcast(&recv_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, socket_comm);
    if (gpu_rank != 0)
    {
        CUDACHECK(cudaIpcOpenMemHandle((void**)&sendbuf_d, send_handle, cudaIpcMemLazyEnablePeerAccess));
        CUDACHECK(cudaIpcOpenMemHandle((void**)&recvbuf_d, recv_handle, cudaIpcMemLazyEnablePeerAccess));
    }

    for (int i = 0; i < max_s_proc; i++)
        sendbuf[i] = ((float)rand()) / RAND_MAX;
    CUDACHECK(cudaMemcpy(&(sendbuf_d[gpu_rank*max_s_proc]), sendbuf, max_s_proc*sizeof(float), cudaMemcpyHostToDevice));

    // NCCL Setup
    ncclUniqueId id;
    if (rank / ppg == 0) ncclGetUniqueId(&id);

    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, gpu_comm);

    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    ncclComm_t nccl_comm;
    NCCLCHECK(ncclCommInitRank(&nccl_comm, num_procs/ppg, id, rank/ppg));

    CUDACHECK(cudaGetLastError());

    print_allreduce(max_p, sendbuf_d, recvbuf_d, recvbuf_std, recvbuf_ipc,
            MPI_COMM_WORLD, nccl_comm, stream, gpu_rank, ppg);

    if (gpu_rank == 0)
    {
        CUDACHECK(cudaFree(sendbuf_d));
        CUDACHECK(cudaFree(recvbuf_d));
    }
    else
    {
        CUDACHECK(cudaIpcCloseMemHandle(sendbuf_d));
        CUDACHECK(cudaIpcCloseMemHandle(recvbuf_d));
    }

    cudaFreeHost(sendbuf);
    cudaFreeHost(recvbuf_std);
    cudaFreeHost(recvbuf_ipc);

    ncclCommDestroy(nccl_comm);

    MPI_Comm_free(&socket_comm);
    MPI_Comm_free(&gpu_comm);
    MPI_Comm_free(&local_comm);


    MPI_Finalize();
    return 0;
}
