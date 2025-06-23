#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>

void compare_allreduce_results(std::vector<double>& pmpi, std::vector<double>& mpix, int s)
{
    for (int j = 0; j < s; j++)
    {
        if (fabs((pmpi[j] - mpix[j])/mpix[j]) > 1e-06)
        {
            fprintf(stderr, "MPIX Allreduce != PMPI, position %d, pmpi %e, mpix %e\n", 
                    j, pmpi[j], mpix[j]);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

double time_allreduce(F allreduce_func, const void* sendbuf, void* recvbuf, int count,
        MPI_Datatype datatype, MPI_Op op, C comm, MPI_Comm global_comm)
{
    double t0, tfinal;
    int n_iters;

    // Warm-Up
    allreduce_func(sendbuf, recvbuf, count, datatype, op, comm);

    // Time 1 iteration
    gpuDeviceSynchronize();
    MPI_Barrier(global_comm);
    t0 = MPI_Wtime();
    allreduce_func(sendbuf, recvbuf, count, datatype, op, comm);
    tfinal = (MPI_Wtime() - t0);
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, global_comm);

    // Estimate N Iterations Needed
    n_iters = 1.0 / t0;

    // Time n_iters iterations
    gpuDeviceSynchronize();
    MPI_Barrier(global_comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
        allreduce_func(sendbuf, recvbuf, count, datatype, op, comm);
    tfinal = (MPI_Wtime() - t0) / n_iters;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, global_comm);

    return t0;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    // Get global rank, num_procs
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Initialize MPIX_Comm, create local communicator
    MPIX_Comm* xcomm;
    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);
    MPIX_Comm_topo_init(xcomm);
    
    // Get local rank, num_procs
    int local_rank, ppn;
    MPI_Comm_rank(xcomm->local_comm, &local_rank);
    MPI_Comm_size(xcomm->local_comm, &ppn);

    // Get local GPU count
    int gpn;
    cudaCheck(cudaGetDeviceCount(&gpn));

    // Get local GPU and rank among GPU procs
    int ppg = ppn / gpn;
    int local_gpu = local_rank / ppg;
    int gpu_rank = local_rank % ppg;
    int max_s_ipc = max_s / ppg;

    // Set local device
    gpuSetDevice(local_rank);

    int max_i = 20;
    int max_s = pow(2, max_i);
    int n_iter = 100;
    double t0, tfinal;
    srand(time(NULL));
    std::vector<double> send_data(max_s);
    std::vector<double> pmpi_allreduce(max_s);
    std::vector<double> mpix_allreduce(max_s);
    for (int j = 0; j < max_s; j++)
        send_data[j] = rand();

    double *send_data_d;
    double *recv_data_d;
    if (gpu_rank == 0)
    {
        gpuMalloc((void**)(&send_data_d), max_s*sizeof(double));
        gpuMalloc((void**)(&recv_data_d), max_s*sizeof(double));
        gpuMemcpy(send_data_d, send_data.data(), max_s*sizeof(double), gpuMemcpyHostToDevice);
    }
    
    MPIX_Comm* xcomm;
    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);
    MPIX_Comm_topo_init(xcomm);
    int local_rank;
    MPI_Comm_rank(xcomm->local_comm, &local_rank);
    gpuSetDevice(local_rank);

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);
        if (rank == 0) printf("Testing Size %d\n", s);

        // Standard MPI Implementation
        gpuMemset(recv_data_d, 0, s*sizeof(int));
        PMPI_Allreduce(send_data_d,
                recv_data_d,
                s,
                MPI_DOUBLE, 
                MPI_SUM,
                MPI_COMM_WORLD);
        gpuMemcpy(pmpi_allreduce.data(), recv_data_d, s*sizeof(double),
                gpuMemcpyDeviceToHost);


        // MPI Advance : GPU-Aware LANE 
        gpuMemset(recv_data_d, 0, s*sizeof(int));
        gpu_aware_allreduce_lane(send_data_d,
                recv_data_d,
                s,
                MPI_DOUBLE, 
                MPI_SUM,
                xcomm);
        gpuMemcpy(mpix_allreduce.data(), recv_data_d, s*sizeof(double),
                gpuMemcpyDeviceToHost);
        compare_allreduce_results(pmpi_allreduce, mpix_allreduce, s);

        // MPI Advance : GPU-Aware LOCALITY 
        gpuMemset(recv_data_d, 0, s*sizeof(int));
        gpu_aware_allreduce_loc(send_data_d,
                recv_data_d,
                s,
                MPI_DOUBLE, 
                MPI_SUM,
                xcomm);
        gpuMemcpy(mpix_allreduce.data(), recv_data_d, s*sizeof(double),
                gpuMemcpyDeviceToHost);
        compare_allreduce_results(pmpi_allreduce, mpix_allreduce, s);


        // Time PMPI Allreduce
        time_allreduce(PMPI_Alltoall, send_data_d, recv_data_d, s,
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Time Lane Allreduce
        time_allreduce(gpu_aware_allreduce_lane, send_data_d, 
                recv_data_d, s, MPI_DOUBLE, MPI_SUM, xcomm);

        // Time Locality-Aware Allreduce
        time_allreduce(gpu_aware_allreduce_loc, send_data_d,
                recv_data_d, s, MPI_DOUBLE, MPI_SUM, xcomm);

    }

    MPIX_Comm_free(&xcomm);

    gpuFree(send_data_d);
    gpuFree(recv_data_d);

    MPI_Finalize();
    return 0;
}
