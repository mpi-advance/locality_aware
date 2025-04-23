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
    }
    CUDACHECK(cudaStreamSynchronize(stream));
    double tfinal = (MPI_Wtime() - t0) / n_iters;
    return tfinal;
}

void print_allreduce(int max_p, float* sendbuf, float* recvbuf, 
        MPI_Comm comm, ncclComm_t nccl_comm, cudaStream_t stream)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (int i = 0; i < max_p; i++)
    { 
        int s = pow(2, i);
        if (rank == 0) printf("Size %d: ", s);
        CUDACHECK(cudaDeviceSynchronize());

        // Warm-Up
        allreduce(s, sendbuf, recvbuf, nccl_comm, stream, comm, 1);

        // Estimate Iterations
        double time = allreduce(s, sendbuf, recvbuf, nccl_comm, stream, comm, 2);
        MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, comm);
        int n_iters = (2.0/time) + 1;

        // Time Allreduce
        time = allreduce(s, sendbuf, recvbuf, nccl_comm, stream, comm, n_iters);
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
    gpuSetDevice(local_gpu);
    MPI_Comm_free(&local_comm);
    
    float* sendbuf_d;
    float* recvbuf_d;

    CUDACHECK(gpuMalloc((void**)&sendbuf_d, max_s*sizeof(float)));
    CUDACHECK(gpuMalloc((void**)&recvbuf_d, max_s*sizeof(float)));

    // NCCL Setup
    if (rank == 0) printf("Getting unique id\n");
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);

    if (rank == 0) printf("Broadcasting unique id\n");
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("Initializing nccl communicator\n");
    ncclComm_t nccl_comm;
    NCCLCHECK(ncclCommInitRank(&nccl_comm, num_procs, id, rank));

    if (rank == 0) printf("Create cuda stream\n");
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    CUDACHECK(cudaGetLastError());

    print_allreduce(max_p, sendbuf_d, recvbuf_d,
            MPI_COMM_WORLD, nccl_comm, stream);

    gpuFree(sendbuf_d);
    gpuFree(recvbuf_d);
    
    if (rank == 0) printf("Destroying nccl comm\n");
    ncclCommDestroy(nccl_comm);

    MPI_Finalize();
    return 0;
}
