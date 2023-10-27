#include "mpi_advance.h"
#include "../src/tests/par_binary_IO.hpp"
#include "../src/tests/sparse_mat.hpp"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>

int main(int argc, char* argv[])
{
    char* filename;
    if (argc <= 1)
    {
        printf("Need Command Line Argument for Filename!\n");
        return -1;
    }
    else
    {
        filename = argv[1]; 
    }

    MPI_Init(&argc, &argv);

    int rank, num_procs;
    long* off_proc_cols_d;
    int off_proc_cols_count;
    int* send_comm_idx_d;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);


    double t0, tfinal; 
    int n_iter = 3; 

    ParMat<int> A;
    readParMatrix(filename, A);
    form_recv_comm(A);
    
    // Copy off proc cols to GPU
    cudaMallocHost((void**)&off_proc_cols_d, sizeof(long)*(A.off_proc_columns.size()));
    cudaMemcpy(off_proc_cols_d, A.off_proc_columns, sizeof(long)*A.off_proc_columns, cudaMemcpyHostToDevice);
    off_proc_cols_count = A.off_proc_columns.size();

    // Test Standard copy to cpu
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        form_send_comm_standard_copy_to_cpu(A, off_proc_cols_d, off_proc_cols_count, send_comm_idx_d);
        A.reset_comm();
        cudaFreeHost(send_comm_idx_d);
    }
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Standard copy to cpu: %e\n", t0);

    // Test Standard gpu aware
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        form_send_comm_standard_gpu_aware(A, off_proc_cols_d, send_comm_idx_d);
        A.reset_comm();
        cudaFreeHost(send_comm_idx_d);
    }
    tfinal = MPI_Wtime() - t0; 
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Standard gpu aware: %e\n", t0);

    // Test Torsten copy to cpu
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_ter; i++)
    {
        form_send_comm_torsten_copy_to_cpu(A, off_proc_cols_d, off_proc_cols_count, send_comm_idx_d);
        cudaFreeHost(send_comm_idx_d);
    }
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) prtinf("Torsten copy to cpu: %e\n", t0);

    cudaFreeHost(off_proc_cols_d);
    MPI_Finalize();
    return 0;
}