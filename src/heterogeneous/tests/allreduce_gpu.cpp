#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>

void allreduce_std(int size, float* sendbuf, float* recvbuf, float* tmpbuf,
        int* recvcounts, MPI_Comm comm, MPI_Comm intra_comm, MPI_Comm inter_comm)
{
    MPI_Allreduce(sendbuf, recvbuf, size, MPI_FLOAT, MPI_SUM, comm);
}

void allreduce_loc(int size, float* sendbuf, float* recvbuf, float* tmpbuf,
        int* recvcounts, MPI_Comm comm, MPI_Comm intra_comm, MPI_Comm inter_comm)
{   
    MPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, MPI_FLOAT,
            MPI_SUM, intra_comm);
    MPI_Allreduce(recvbuf, tmpbuf, recvcounts[0], MPI_FLOAT,
            MPI_SUM, inter_comm);
    MPI_Allgather(tmpbuf, recvcounts[0], MPI_FLOAT, recvbuf, 
            recvcounts[0], MPI_FLOAT, intra_comm);
}

void allreduce_lane(int size, float* sendbuf, float* recvbuf, float* tmpbuf,
        int* recvcounts, MPI_Comm comm, MPI_Comm intra_comm, MPI_Comm inter_comm)
{   
    MPI_Allreduce(sendbuf, tmpbuf, size, MPI_FLOAT,
            MPI_SUM, inter_comm);
    MPI_Allreduce(tmpbuf, recvbuf, size, MPI_FLOAT,
            MPI_SUM, intra_comm);
}

template <typename F>
double allreduce_timer(F allreduce, int size, float* sendbuf, float* recvbuf, 
        float* tmpbuf, int* recvcounts, MPI_Comm comm, MPI_Comm intra_comm, 
        MPI_Comm inter_comm, int n_iters)
{
    gpuDeviceSynchronize();
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        allreduce(size, sendbuf, recvbuf, tmpbuf, recvcounts,
                comm, intra_comm, inter_comm);
    }
    double tfinal = (MPI_Wtime() - t0) / n_iters;
    return tfinal;
}

template <typename F>
int estimate_iters(F allreduce, int size, float* sendbuf, float* recvbuf,
        float* tmpbuf, int* recvcounts, MPI_Comm comm, MPI_Comm intra_comm,
        MPI_Comm inter_comm)
{
    // Warm-Up
    allreduce_timer(allreduce, size, sendbuf, recvbuf, tmpbuf, recvcounts,
            comm, intra_comm, inter_comm, 1);

    // Time 2 iterations
    double time = allreduce_timer(allreduce, size, sendbuf, recvbuf, tmpbuf, recvcounts,
            comm, intra_comm, inter_comm, 2);
    
    MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, comm);
    return (1.0 / time) + 1;
}

template <typename F>
double allreduce_test(F allreduce, int size, float* sendbuf, float* recvbuf,
        float* tmpbuf, int* recvcounts, MPI_Comm comm, MPI_Comm intra_comm,
        MPI_Comm inter_comm)
{
    int n_iters = estimate_iters(allreduce, size, sendbuf, recvbuf, tmpbuf,
            recvcounts, comm, intra_comm, inter_comm);

    return allreduce_timer(allreduce, size, sendbuf, recvbuf, tmpbuf,
            recvcounts, comm, intra_comm, inter_comm, n_iters);
}

void print_allreduce(int max_p, float* sendbuf, float* recvbuf, float* tmpbuf,
        float* recvbuf_std, float* recvbuf_new,
        MPI_Comm comm, MPI_Comm intra_comm, MPI_Comm inter_comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    double time; 
    int* recvcounts = new int[num_procs];

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    int start_i = log2(num_gpus);
    for (int i = start_i; i < max_p; i++)
    { 
        int s = pow(2, i);
        if (rank == 0) printf("Size %d\n", s);

        for (int i = 0; i < num_procs; i++)
            recvcounts[i] = s / num_gpus;

        /*************************
        *** Test for Correctness
        *************************/
        // Copy standard recvbuf to recvbuf_std
        cudaMemset(recvbuf, 0, s*sizeof(float));
        allreduce_std(s, sendbuf, recvbuf, tmpbuf, recvcounts, comm, 
                intra_comm, inter_comm);
        cudaMemcpy(recvbuf_std, recvbuf, s*sizeof(float), cudaMemcpyDeviceToHost);

        // Copy loc results to recvbuf_new
        cudaMemset(recvbuf, 0, s*sizeof(float));
        allreduce_loc(s, sendbuf, recvbuf, tmpbuf, recvcounts, comm, 
                intra_comm, inter_comm);
        cudaMemcpy(recvbuf_new, recvbuf, s*sizeof(float), cudaMemcpyDeviceToHost);
        float maxError = 0.0;
        // Compare recvbuf std and recvbuf new
        for (int i = 0; i < s; i++)
        {
            if (fabs(recvbuf_new[i] - recvbuf_std[i]) > maxError) maxError = fabs(recvbuf_new[i] - recvbuf_std[i]);
            if (fabs(recvbuf_new[i] - recvbuf_std[i]) > 1e-3)
            {
                printf("DIFFERENCE IN RESULTS in LOC! %e vs %e\n", recvbuf_new[i], recvbuf_std[i]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &maxError, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0) printf("LOC MAX ERROR: %e\n", maxError);

        // Copy lane results to recvbuf_new
        cudaMemset(recvbuf, 0, s*sizeof(float));
        allreduce_lane(s, sendbuf, recvbuf, tmpbuf, recvcounts, comm,
                intra_comm, inter_comm);
        cudaMemcpy(recvbuf_new, recvbuf, s*sizeof(float), cudaMemcpyDeviceToHost);
        maxError = 0.0;
        // Compare recvbuf std and recvbuf new
        for (int i = 0; i < s; i++)
        {
            if (fabs(recvbuf_new[i] - recvbuf_std[i]) > maxError) maxError = fabs(recvbuf_new[i] - recvbuf_std[i]);
            if (fabs(recvbuf_new[i] - recvbuf_std[i]) > 1e-3)
            {
                printf("DIFFERENCE IN RESULTS in LANE! %e vs %e\n", recvbuf_new[i], recvbuf_std[i]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
	MPI_Allreduce(MPI_IN_PLACE, &maxError, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0) printf("LANE MAX ERROR: %e\n", maxError);

        /*************************
        *** Time Methods
        *************************/
        time = allreduce_test(allreduce_std, s, sendbuf, recvbuf, tmpbuf,
                recvcounts, comm, intra_comm, inter_comm);
        if (rank == 0) printf("STD: %e\n", time);

        time = allreduce_test(allreduce_loc, s, sendbuf, recvbuf, tmpbuf,
                recvcounts, comm, intra_comm, inter_comm);
        if (rank == 0) printf("LOC: %e\n", time);
        
        time = allreduce_test(allreduce_lane, s, sendbuf, recvbuf, tmpbuf,
                recvcounts, comm, intra_comm, inter_comm);
        if (rank == 0) printf("LANE: %e\n", time);

        fflush(stdout);
    }
    if (rank == 0) printf("\n");

    delete[] recvcounts;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int max_p = 28;
    int max_s = pow(2, max_p - 1);

    // Set Local GPU
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 
            rank, MPI_INFO_NULL, &local_comm);
    int local_rank, ppn;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &ppn);

    MPI_Comm inter_comm;
    MPI_Comm_split(MPI_COMM_WORLD, local_rank, rank, &inter_comm);

    int gpn;
    gpuGetDeviceCount(&gpn);

    int ppg = ppn / gpn;
    int local_gpu = local_rank / ppg;
    if (argc < 2 || (strcmp(argv[1], "r") != 0))
    {
        gpuSetDevice(local_gpu);
    }
    else
    {
        gpuSetDevice((gpn - local_gpu) - 1);
        printf("rank %d reversed\n", rank);
        fflush(stdout);
    }

    float* sendbuf;
    float* recvbuf;
    float* tmpbuf;

    float* sendbuf_h;
    float* recvbuf_std;
    float* recvbuf_new;

    gpuMalloc((void**)&sendbuf, max_s*sizeof(float));
    gpuMalloc((void**)&recvbuf, max_s*sizeof(float));
    gpuMalloc((void**)&tmpbuf, max_s*sizeof(float));
    
    gpuMallocHost((void**)&sendbuf_h, max_s*sizeof(float));
    gpuMallocHost((void**)&recvbuf_std, max_s*sizeof(float));
    gpuMallocHost((void**)&recvbuf_new, max_s*sizeof(float));

    srand(time(NULL) + (rank * 120));
    for (int i = 0; i < max_s; i++)
        sendbuf_h[i] = ((float)rand()) / RAND_MAX;
    gpuMemcpy(sendbuf, sendbuf_h, max_s*sizeof(float), gpuMemcpyHostToDevice);
    gpuDeviceSynchronize();
    
    print_allreduce(max_p, sendbuf, recvbuf, tmpbuf, recvbuf_std,
             recvbuf_new, MPI_COMM_WORLD, local_comm, inter_comm);

    gpuFreeHost(sendbuf_h);
    gpuFreeHost(recvbuf_std);
    gpuFreeHost(recvbuf_new);

    gpuFree(sendbuf);
    gpuFree(recvbuf);
    gpuFree(tmpbuf);

    MPI_Comm_free(&local_comm);
    MPI_Comm_free(&inter_comm);

    MPI_Finalize();
    return 0;
}
