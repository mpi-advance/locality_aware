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

void allreduce_std(int size, float* sendbuf, float* recvbuf, float* tmpbuf,
        ncclComm_t nccl_comm, ncclComm_t intra_comm, ncclComm_t inter_comm, 
        cudaStream_t stream)
{
    NCCLCHECK(ncclAllReduce((const void*) sendbuf, (void*)recvbuf, size, ncclFloat,
            ncclSum, nccl_comm, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
}

void allreduce_loc(int size, float* sendbuf, float* recvbuf, float* tmpbuf,
        ncclComm_t nccl_comm, ncclComm_t intra_comm, ncclComm_t inter_comm, 
        cudaStream_t stream)
{
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    ncclReduceScatter(sendbuf, recvbuf, size/num_gpus, ncclFloat,
            ncclSum, intra_comm, stream);
    ncclAllReduce(recvbuf, tmpbuf, size/num_gpus, ncclFloat,
            ncclSum, inter_comm, stream);
    ncclAllGather(tmpbuf, recvbuf, size/num_gpus, ncclFloat,
            intra_comm, stream);
    CUDACHECK(cudaStreamSynchronize(stream));
}

void allreduce_lane(int size, float* sendbuf, float* recvbuf, float* tmpbuf,
        ncclComm_t nccl_comm, ncclComm_t intra_comm, ncclComm_t inter_comm,
        cudaStream_t stream)
{   
    ncclAllReduce(sendbuf, tmpbuf, size, ncclFloat,
            ncclSum, inter_comm, stream);
    ncclAllReduce(tmpbuf, recvbuf, size, ncclFloat,
            ncclSum, intra_comm, stream);
    CUDACHECK(cudaStreamSynchronize(stream));
}

template <typename F>
double allreduce_timer(F allreduce, int size, float* sendbuf, float* recvbuf, float* tmpbuf,
        ncclComm_t nccl_comm, ncclComm_t intra_comm, ncclComm_t inter_comm,
        cudaStream_t stream, MPI_Comm comm, int n_iters)
{
    cudaStreamSynchronize(stream);
    gpuDeviceSynchronize();
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        allreduce(size, sendbuf, recvbuf, tmpbuf, nccl_comm, 
                intra_comm, inter_comm, stream);
    }
    double tfinal = (MPI_Wtime() - t0) / n_iters;
    return tfinal;
}

template <typename F>
int estimate_iters(F allreduce, int size, float* sendbuf, float* recvbuf, float* tmpbuf,
        ncclComm_t nccl_comm, ncclComm_t intra_comm, ncclComm_t inter_comm,
        cudaStream_t stream, MPI_Comm comm)
{
    // Warm-Up
    allreduce_timer(allreduce, size, sendbuf, recvbuf, tmpbuf, 
            nccl_comm, intra_comm, inter_comm, stream, comm, 1);

    // Time 2 iterations
    double time = allreduce_timer(allreduce, size, sendbuf, recvbuf, tmpbuf, 
            nccl_comm, intra_comm, inter_comm, stream, comm, 2);

    MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, comm);
    return (1.0/time) + 1;
}

template <typename F>
double allreduce_test(F allreduce, int size, float* sendbuf, float* recvbuf, float* tmpbuf,
        ncclComm_t nccl_comm, ncclComm_t intra_comm, ncclComm_t inter_comm,
        cudaStream_t stream, MPI_Comm comm)
{
    int n_iters = estimate_iters(allreduce, size, sendbuf, recvbuf, tmpbuf,
            nccl_comm, intra_comm, inter_comm, stream, comm);
    return allreduce_timer(allreduce, size, sendbuf, recvbuf, tmpbuf,
            nccl_comm, intra_comm, inter_comm, stream, comm, n_iters);
}

void print_allreduce(int max_p, float* sendbuf, float* recvbuf, float* tmpbuf, 
        float* recvbuf_std, float* recvbuf_new,
        MPI_Comm comm, ncclComm_t nccl_comm, ncclComm_t inter_comm,
        ncclComm_t intra_comm, cudaStream_t stream)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double time;

    for (int i = 0; i < max_p; i++)
    { 
        int s = pow(2, i);
        if (rank == 0) printf("Size %d\n", s);
        CUDACHECK(cudaDeviceSynchronize());


        /************************
        *** Check for Correctness
        ************************/
        // Copy standard result to recvbuf_std
        cudaMemset(recvbuf, 0, s*sizeof(float));
        allreduce_std(s, sendbuf, recvbuf, tmpbuf, nccl_comm,
                intra_comm, inter_comm, stream);
        cudaMemcpy(recvbuf_std, recvbuf, s*sizeof(float), cudaMemcpyDeviceToHost);

        // Copy loc result to recvbuf new
        cudaMemset(recvbuf, 0, s*sizeof(float));
        allreduce_loc(s, sendbuf, recvbuf, tmpbuf, nccl_comm,
                intra_comm, inter_comm, stream);

        // Compare recvbuf std and recvbuf new
        cudaMemcpy(recvbuf_new, recvbuf, s*sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < s; i++)
            if (fabs(recvbuf_new[i] - recvbuf_std[i]) > 1e-6)
            {
                printf("DIFFERENCE IN RESULTS! %e vs %e\n", recvbuf_new[i], recvbuf_std[i]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

        // Copy lane result to recvbuf new
        cudaMemset(recvbuf, 0, s*sizeof(float));
        allreduce_lane(s, sendbuf, recvbuf, tmpbuf, nccl_comm,
                intra_comm, inter_comm, stream);
        cudaMemcpy(recvbuf_new, recvbuf, s*sizeof(float), cudaMemcpyDeviceToHost);
 
        // Compare recvbuf std and recvbuf new
        for (int i = 0; i < s; i++)
            if (fabs(recvbuf_new[i] - recvbuf_std[i]) > 1e-6)
            {
                printf("DIFFERENCE IN RESULTS! %e vs %e\n", recvbuf_new[i], recvbuf_std[i]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }



        /************************
        *** Time Methods
        ************************/
        time = allreduce_test(allreduce_std, s, sendbuf, recvbuf, tmpbuf, nccl_comm,
                intra_comm, inter_comm, stream, comm);
        if (rank == 0) printf("STD: %e\n", time);

        time = allreduce_test(allreduce_loc, s, sendbuf, recvbuf, tmpbuf, nccl_comm,
                intra_comm, inter_comm, stream, comm);
        if (rank == 0) printf("LOC: %e\n", time);

        time = allreduce_test(allreduce_lane, s, sendbuf, recvbuf, tmpbuf, nccl_comm,
                intra_comm, inter_comm, stream, comm);
        if (rank == 0) printf("LANE: %e\n", time);

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

    MPI_Comm gpu_comm;
    MPI_Comm_split(MPI_COMM_WORLD, local_gpu, rank, &gpu_comm);

    float* sendbuf_d;
    float* recvbuf_d;
    float* tmpbuf_d;

    CUDACHECK(gpuMalloc((void**)&sendbuf_d, max_s*sizeof(float)));
    CUDACHECK(gpuMalloc((void**)&recvbuf_d, max_s*sizeof(float)));
    CUDACHECK(gpuMalloc((void**)&tmpbuf_d, max_s*sizeof(float)));

    float *recvbuf_std, *recvbuf_new;
    CUDACHECK(gpuMallocHost((void**)&recvbuf_std, max_s*sizeof(float)));
    CUDACHECK(gpuMallocHost((void**)&recvbuf_new, max_s*sizeof(float)));


    // NCCL Setup
    if (rank == 0) printf("Getting unique id\n");
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);

    if (rank == 0) printf("Broadcasting unique id\n");
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("Initializing nccl communicator\n");
    ncclComm_t nccl_comm;
    NCCLCHECK(ncclCommInitRank(&nccl_comm, num_procs, id, rank));

    if (rank == 0) printf("Splitting for intra node communicator\n");
    ncclComm_t intra_comm;
    ncclCommSplit(nccl_comm, rank / ppn, local_rank, &intra_comm, NULL);
 
    if (rank == 0) printf("Splitting for inter node communicator\n");
    ncclComm_t inter_comm;
    ncclCommSplit(nccl_comm, local_rank, rank / ppn, &inter_comm, NULL);

    if (rank == 0) printf("Create cuda stream\n");
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    CUDACHECK(cudaGetLastError());

    print_allreduce(max_p, sendbuf_d, recvbuf_d, tmpbuf_d, recvbuf_std, recvbuf_new,
            MPI_COMM_WORLD, nccl_comm, inter_comm, intra_comm, stream);

    gpuFreeHost(recvbuf_std);
    gpuFreeHost(recvbuf_new);

    gpuFree(sendbuf_d);
    gpuFree(recvbuf_d);
    gpuFree(tmpbuf_d);
    
    if (rank == 0) printf("Destroying nccl comm\n");
    ncclCommDestroy(nccl_comm);

    MPI_Comm_free(&gpu_comm);
    MPI_Comm_free(&local_comm);
    
    MPI_Finalize();
    return 0;
}
