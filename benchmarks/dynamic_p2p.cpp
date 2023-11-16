#ifndef PROFILE
#define PROFILE

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
    int size;
    if (argc <= 2)
    {
        printf("Need Command Line Arg for Density of CommPkg and Size!\n");
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
	size = atoi(argv[2]);
    }

    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    MPIX_Comm* xcomm;
    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);

    double t0, tfinal;
    
    int n_msgs = num_procs * density;
    int n_iter = 10000;
    if (n_msgs > 100 || size > 1000)
        n_iter = 1000;
    if (n_msgs > 1000 || size > 10000)
        n_iter = 10;

    ParMat<int> A;
    A.recv_comm->n_msgs = n_msgs;
    A.recv_comm->procs.resize(n_msgs);
    A.recv_comm->ptr.resize(n_msgs+1);
    A.recv_comm->req.resize(n_msgs);
    A.recv_comm->counts.resize(n_msgs);
    A.recv_comm->size_msgs = n_msgs * size; // each msg same size
    A.off_proc_num_cols = A.recv_comm->size_msgs;
    A.off_proc_columns.resize(A.recv_comm->size_msgs);
    A.first_col = rank * size;

    A.recv_comm->ptr[0] = 0;
    int ctr, dist;
    ctr = 0;
    for (int i = 0; i < num_procs; i++)
    {
        dist = i - rank;
	if (dist <= 0) dist += num_procs;
	if (dist > n_msgs) continue;

        A.recv_comm->procs[ctr] = i;
        A.recv_comm->ptr[ctr+1] = A.recv_comm->ptr[ctr] + size;
        A.recv_comm->counts[ctr] = size;
	for (int j = 0; j < size; j++)
            A.off_proc_columns[ctr*size+j] = i*size + j;
	ctr++;
    }
    std::vector<int> send_data(A.off_proc_num_cols);
    std::vector<int> recv_data_std(A.recv_comm->size_msgs);
    std::vector<int> recv_data(A.recv_comm->size_msgs);
    for (int i = 0; i < A.recv_comm->size_msgs; i++)
    {
        send_data[i] = i;
    }

    int max_n = A.recv_comm->n_msgs;
    MPI_Allreduce(MPI_IN_PLACE, &max_n, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("Max N Msgs : %d\n", max_n);

    #undef PROFILE
    // Check for Correctness
    MPI_Barrier(MPI_COMM_WORLD);
    form_send_comm_standard(A);
    communicate(A, send_data, recv_data_std, MPI_INT);
    A.reset_comm();

    MPI_Barrier(MPI_COMM_WORLD);
    form_send_comm_torsten(A);
    communicate(A, send_data, recv_data, MPI_INT);
    if (rank == 0) for (int i = 0; i < A.recv_comm->size_msgs; i++)
	    if (recv_data_std[i] != recv_data[i])
		    printf("i %d, std %d, new %d\n", i, recv_data_std[i], recv_data[i]);
    A.reset_comm();

    MPI_Barrier(MPI_COMM_WORLD);
    form_send_comm_torsten_loc(A, xcomm);
    communicate(A, send_data, recv_data, MPI_INT);
    if (rank == 0) for (int i = 0; i < A.recv_comm->size_msgs; i++)
	    if (recv_data_std[i] != recv_data[i])
		    printf("i %d, std %d, new %d\n", i, recv_data_std[i], recv_data[i]);
    A.reset_comm();

    int flag;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
    if (flag) printf("Rank %d has hanging msg\n", rank);
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, xcomm->local_comm, &flag, MPI_STATUS_IGNORE);
    if (flag) printf("Rank %d has hanging lcl msg\n", rank);
    #define PROFILE

    // Test Standard (all reduce + dynamic p2p)
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < n_iter; i++)
    {
        form_send_comm_standard(A);
	      A.reset_comm();
    }

    // Test Torsten (dynamic p2p)
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < n_iter; i++)
    {
        form_send_comm_torsten(A);
	      A.reset_comm();
    }

    // Test communicate
    for (int i = 0; i < n_iter; i++)
    {
      communicate(A, send_data, recv_data, MPI_INT);
      send_data.clear();
      recv_data.clear();
    }

    MPIX_Comm_free(xcomm);
    MPI_Finalize();
    return 0;
}