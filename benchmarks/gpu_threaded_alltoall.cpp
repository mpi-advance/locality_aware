#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>
#include <omp.h>

int main(int argc, char* argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    omp_set_num_threads(10);

    int max_i = 20;
    int max_s = pow(2, max_i);
    int max_n_iter = 100;
    double t0, tfinal;
    srand(time(NULL));
    std::vector<double> send_data(max_s*num_procs);
    std::vector<double> pmpi_alltoall(max_s*num_procs);
    std::vector<double> mpix_alltoall(max_s*num_procs);
    for (int j = 0; j < max_s*num_procs; j++)
        send_data[j] = rand();

    double* send_data_d;
    double* recv_data_d;
    cudaMalloc((void**)(&send_data_d), max_s*num_procs*sizeof(double));
    cudaMalloc((void**)(&recv_data_d), max_s*num_procs*sizeof(double));
    cudaMemcpy(send_data_d, send_data.data(), max_s*num_procs*sizeof(double), cudaMemcpyHostToDevice);

    MPIX_Comm* locality_comm;
    MPIX_Comm_init(&locality_comm, MPI_COMM_WORLD);

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);
        if (rank == 0) printf("Testing Size %d\n", s);

        int n_iter = max_n_iter;
        if (s > 4096) n_iter /= 10;

        // Standard MPI Implementation
        PMPI_Alltoall(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                MPI_COMM_WORLD);
        cudaMemcpy(pmpi_alltoall.data(), recv_data_d, s*num_procs*sizeof(double),
                cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, s*num_procs*sizeof(int));

        // MPI Advance : Threaded Pairwise Exchange
        threaded_alltoall_pairwise(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                locality_comm);
        cudaMemcpy(mpix_alltoall.data(), recv_data_d, s*num_procs*sizeof(double),
                cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, s*num_procs*sizeof(int));
        for (int j = 0; j < s; j++)
	{
            if (fabs(pmpi_alltoall[j] - mpix_alltoall[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, pmpi %e, Thread-PE %e\n", 
                         rank, j, pmpi_alltoall[j], mpix_alltoall[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

        // MPI Advance : Copy To CPU Nonblocking (P2P)
        threaded_alltoall_nonblocking(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                locality_comm);
        cudaMemcpy(mpix_alltoall.data(), recv_data_d, s*num_procs*sizeof(double),
                cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, s*num_procs*sizeof(int));
        for (int j = 0; j < s; j++)
	{
            if (fabs(pmpi_alltoall[j] - mpix_alltoall[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, pmpi %e, Thread-NB %e\n", 
                         rank, j, pmpi_alltoall[j], mpix_alltoall[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }


        // Time PMPI Alltoall
        PMPI_Alltoall(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                MPI_COMM_WORLD);
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

        // Time Threaded Pairwise Exchange
        threaded_alltoall_pairwise(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                locality_comm);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
            threaded_alltoall_pairwise(send_data_d,
                    s,
                    MPI_DOUBLE, 
                    recv_data_d,
                    s,
                    MPI_DOUBLE,
                    locality_comm);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Threaded Pairwise Exchange Time %e\n", t0);

        // Time Threaded Nonblocking
        threaded_alltoall_nonblocking(send_data_d,
                s,
                MPI_DOUBLE, 
                recv_data_d,
                s,
                MPI_DOUBLE,
                locality_comm);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
            threaded_alltoall_nonblocking(send_data_d,
                    s,
                    MPI_DOUBLE, 
                    recv_data_d,
                    s,
                    MPI_DOUBLE,
                    locality_comm);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Threaded Nonblocking Time %e\n", t0);
    }
    cudaFree(send_data_d);
    cudaFree(recv_data_d);

    MPIX_Comm_free(locality_comm);

    MPI_Finalize();
    return 0;
}
