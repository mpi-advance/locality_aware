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
    double density;
    if (argc <= 1)
    {
        printf("Need Command Line Arg for Density of CommPkg!\n");
        return -1;
    }
    else
    {
        density = atof(argv[1]);
        if (density < 0  || density > 1)
        {
            printf("Density should be floating point number between 0 and 1\n");
            return -1;
        }
    }

    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double t0, tfinal;
    int n_iter = 10000;

    int n_msgs = num_procs * density;
    ParMat<int> A;
    A.recv_comm->n_msgs = n_msgs;
    A.recv_comm->procs.resize(n_msgs);
    A.recv_comm->ptr.resize(n_msgs+1);
    A.recv_comm->req.resize(n_msgs);
    A.recv_comm->counts.resize(n_msgs);
    A.recv_comm->size_msgs = n_msgs; // each msg size 1
    A.off_proc_num_cols = n_msgs;
    A.off_proc_columns.resize(n_msgs);

    A.recv_comm->ptr[0] = 0;
    for (int i = 0; i < n_msgs; i++)
    {
        int proc = rank + i;
        if (proc >= num_procs)
            proc -= num_procs;
        A.recv_comm->procs[i] = proc;
        A.recv_comm->ptr[i+1] = A.recv_comm->ptr[i] + 1;
        A.recv_comm->counts[i] = 1;
	A.off_proc_columns[i] = i;
    }

    MPI_Win win;
    int* sizes;
    t0 = MPI_Wtime();
    allocate_rma_dynamic(&win, &sizes);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Window creation : %e\n", t0);
    form_send_comm_rma_dynamic(A, win, sizes);
    int max_n = A.send_comm->n_msgs;
    MPI_Allreduce(MPI_IN_PLACE, &max_n, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("Max N Msgs : %d\n", max_n);
    A.reset_comm();

    // Test Standard
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        form_send_comm_standard(A);
	A.reset_comm();
    }
    tfinal = (MPI_Wtime() - t0) / n_iter;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Standard form comm : %e\n", t0);

    // Test Torsten
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        form_send_comm_torsten(A);
	A.reset_comm();
    }
    tfinal = (MPI_Wtime() - t0) / n_iter;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Torsten form comm : %e\n", t0);

    // Test RMA
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        form_send_comm_rma_dynamic(A, win, sizes);
	A.reset_comm();
    }
    tfinal = (MPI_Wtime() - t0) / n_iter;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("RMA form comm : %e\n", t0);

    free_rma_dynamic(&win, sizes);

    MPI_Finalize();
    return 0;
}
