#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int max_i = 20;
    int max_s = pow(2, max_i);
    int n_iter = 100;
    double t0, tfinal;
    srand(time(NULL));
    std::vector<double> send_data(max_s*num_procs);
    std::vector<double> pmpi_alltoall(max_s*num_procs);
    std::vector<double> mpix_alltoall(max_s*num_procs);
    for (int j = 0; j < max_s*num_procs; j++)
        send_data[j] = rand();

    double* send_data_d;
    double* recv_data_d;
    gpuMalloc((void**)(&send_data_d), max_s*num_procs*sizeof(double));
    gpuMalloc((void**)(&recv_data_d), max_s*num_procs*sizeof(double));
    gpuMemcpy(send_data_d, send_data.data(), max_s*num_procs*sizeof(double), gpuMemcpyHostToDevice);

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
        PMPI_Alltoall(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                MPI_COMM_WORLD);
        gpuMemcpy(pmpi_alltoall.data(), recv_data_d, s*num_procs*sizeof(double),
                gpuMemcpyDeviceToHost);
        gpuMemset(recv_data_d, 0, s*num_procs*sizeof(int));


        // MPI Advance : GPU-Aware Pairwise Exchange
        gpu_aware_alltoall_pairwise(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                xcomm);
        gpuMemcpy(mpix_alltoall.data(), recv_data_d, s*num_procs*sizeof(double),
                gpuMemcpyDeviceToHost);
        gpuMemset(recv_data_d, 0, s*num_procs*sizeof(int));
        for (int j = 0; j < s; j++)
	{
            if (fabs(pmpi_alltoall[j] - mpix_alltoall[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, pmpi %e, GA-PE %e\n", 
                         rank, j, pmpi_alltoall[j], mpix_alltoall[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

        // MPI Advance : GPU-Aware Nonblocking (P2P)
        gpu_aware_alltoall_nonblocking(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                xcomm);
        gpuMemcpy(mpix_alltoall.data(), recv_data_d, s*num_procs*sizeof(double),
                gpuMemcpyDeviceToHost);
        gpuMemset(recv_data_d, 0, s*num_procs*sizeof(int));
        for (int j = 0; j < s; j++)
	{
            if (fabs(pmpi_alltoall[j] - mpix_alltoall[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, pmpi %e, GA-NB %e\n", 
                         rank, j, pmpi_alltoall[j], mpix_alltoall[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

        // MPI Advance : Copy to CPU Pairwise Exchange
        copy_to_cpu_alltoall_pairwise(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                xcomm);
        gpuMemcpy(mpix_alltoall.data(), recv_data_d, s*num_procs*sizeof(double),
                gpuMemcpyDeviceToHost);
        gpuMemset(recv_data_d, 0, s*num_procs*sizeof(int));
        for (int j = 0; j < s; j++)
	{
            if (fabs(pmpi_alltoall[j] - mpix_alltoall[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, pmpi %e, C2C-PE %e\n", 
                         rank, j, pmpi_alltoall[j], mpix_alltoall[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

        // MPI Advance : Copy To CPU Nonblocking (P2P)
        copy_to_cpu_alltoall_nonblocking(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                xcomm);
        gpuMemcpy(mpix_alltoall.data(), recv_data_d, s*num_procs*sizeof(double),
                gpuMemcpyDeviceToHost);
        gpuMemset(recv_data_d, 0, s*num_procs*sizeof(int));
        for (int j = 0; j < s; j++)
	{
            if (fabs(pmpi_alltoall[j] - mpix_alltoall[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, pmpi %e, C2C-NB %e\n", 
                         rank, j, pmpi_alltoall[j], mpix_alltoall[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }




        // Time PMPI Alltoall
        // 1. Warm Up
        PMPI_Alltoall(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                MPI_COMM_WORLD);

        // 2. Calculate n_iter (tfinal ~ 1 sec)
        gpuDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        PMPI_Alltoall(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                MPI_COMM_WORLD);
        tfinal = MPI_Wtime() - t0;
        MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (t0 >= 1.0)
            n_iter = 1;
        else
            n_iter = (1.0 / tfinal);

        // 3. Measure Timing
        gpuDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
            PMPI_Alltoall(send_data_d,
                    s,
                    MPI_DOUBLE, 
                    recv_data_d,
                    s,
                    MPI_DOUBLE,
                    MPI_COMM_WORLD);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("PMPI_Alltoall Time %e\n", t0);


        // Time GPU-Aware Pairwise Exchange
        // 1. Warm Up
        gpu_aware_alltoall_pairwise(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                xcomm);

        // 2. Calculate n_iter (tfinal ~ 1 sec)
        gpuDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        gpu_aware_alltoall_pairwise(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                xcomm);
        tfinal = MPI_Wtime() - t0;
        MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (t0 >= 1.0)
            n_iter = 1;
        else
            n_iter = (1.0 / tfinal);

        // 3. Measure Timing
        gpuDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
            gpu_aware_alltoall_pairwise(send_data_d,
                    s,
                    MPI_DOUBLE, 
                    recv_data_d,
                    s,
                    MPI_DOUBLE,
                    xcomm);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("GPU-Aware Pairwise Exchange Time %e\n", t0);


        // Time GPU-Aware Nonblocking
        // 1. Warm-Up
        gpu_aware_alltoall_nonblocking(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                xcomm);

        // 2. Calculate n_iter (tfinal ~ 1 sec)
        gpuDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        gpu_aware_alltoall_nonblocking(send_data_d,
                s,
                MPI_DOUBLE,
                recv_data_d,
                s,
                MPI_DOUBLE,
                xcomm);
        tfinal = MPI_Wtime() - t0;
        MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (t0 >= 1.0)
            n_iter = 1;
        else
            n_iter = (1.0 / tfinal);

        // 3. Measure Timing
        gpuDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
            gpu_aware_alltoall_nonblocking(send_data_d,
                    s,
                    MPI_DOUBLE, 
                    recv_data_d,
                    s,
                    MPI_DOUBLE,
                    xcomm);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("GPU-Aware Nonblocking Time %e\n", t0);


        // Time Copy-to-CPU Pairwise Exchange
        // 1. Warm-Up
        copy_to_cpu_alltoall_pairwise(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                xcomm);
        gpuDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        // 2. Calculate n_iter (tfinal ~ 1 sec)
        gpuDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        copy_to_cpu_alltoall_pairwise(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                xcomm);
        tfinal = MPI_Wtime() - t0;
        MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (t0 >= 1.0)
            n_iter = 1;
        else
            n_iter = (1.0 / tfinal);

        // 3. Measure Timing
        gpuDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
            copy_to_cpu_alltoall_pairwise(send_data_d,
                    s,
                    MPI_DOUBLE, 
                    recv_data_d,
                    s,
                    MPI_DOUBLE,
                    xcomm);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Copy-to-CPU Pairwise Exchange Time %e\n", t0);


        // Time Copy-to-CPU Nonblocking
        // 1. Warm-Up
        copy_to_cpu_alltoall_nonblocking(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                xcomm);

        // 2. Calculate n_iter (tfinal ~ 1 sec)
        gpuDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        copy_to_cpu_alltoall_nonblocking(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                xcomm);
        tfinal = MPI_Wtime() - t0;
        MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (t0 >= 1.0)
            n_iter = 1;
        else
            n_iter = (1.0 / tfinal);

        // 3. Measure Timing
        gpuDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
            copy_to_cpu_alltoall_nonblocking(send_data_d,
                    s,
                    MPI_DOUBLE, 
                    recv_data_d,
                    s,
                    MPI_DOUBLE,
                    xcomm);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Copy-to-CPU Nonblocking Time %e\n", t0);
    
    }

    MPIX_Comm_free(&xcomm);

    gpuFree(send_data_d);
    gpuFree(recv_data_d);

    MPI_Finalize();
    return 0;
}
