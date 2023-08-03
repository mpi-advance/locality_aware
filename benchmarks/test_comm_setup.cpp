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
        printf("Need Command Line Arg for Filename!\n");
        return -1;
    }
    else
    {
        filename = argv[1];
    }

    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double t0, tfinal;
    int n_iter = 100;

    MPI_Win win;
    int* sizes;
    allocate_rma_dynamic(&win, &sizes);

    // Test Standard
    for (int i = 0; i < n_iter; i++)
    {
        ParMat<int> A;
        readParMatrix(filename, A);
        form_recv_comm(A);
    t0 = MPI_Wtime();
        form_send_comm_standard(A);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Iter %d, Standard form comm : %e\n", i, t0);
    }

    // Test Torsten
    for (int i = 0; i < n_iter; i++)
    {
        ParMat<int> A;
        readParMatrix(filename, A);
        form_recv_comm(A);
    t0 = MPI_Wtime();
        form_send_comm_torsten(A);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Iter %d, Torsten form comm : %e\n", i, t0);
    }

    // Test RMA
    for (int i = 0; i < n_iter; i++)
    {
        ParMat<int> A;
        readParMatrix(filename, A);
        form_recv_comm(A);
    t0 = MPI_Wtime();
        form_send_comm_rma_dynamic(A, win, sizes);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Iter %d, RMA form comm : %e\n", i, t0);
    }



    free_rma_dynamic(&win, sizes);

    MPI_Finalize();
    return 0;
}
