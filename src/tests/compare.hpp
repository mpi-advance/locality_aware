#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <numeric>
#include <set>

void compare(int n_recvs, int std_n_recvs, std::vector<int>& src,
    std::vector<int>& recvvals, std::vector<int>& std_recvvals)
{
    if (n_recvs != std_n_recvs)
    {
        fprintf(stderr, "Incorrect NRecvs! New %d, Std %d\n", n_recvs, std_n_recvs);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    for (int i = 0; i < n_recvs; i++)
    {
        if (recvvals[i] != std_recvvals[src[i]])
        {
            fprintf(stderr, "Recv %d, RecvVals New %d, Std %d\n", i, 
                    recvvals[i], std_recvvals[src[i]]);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void compare(int n_recvs, int std_n_recvs, int s_recvs, int std_s_recvs,
    std::vector<int>& src, std::vector<int>& recvcounts,
    std::vector<int>& rdispls, std::vector<long>& recvvals,
    int first_col, std::vector<int>& proc_counts,
    std::vector<int>& proc_displs, std::vector<int>& indices)
{
    int proc;
    if (n_recvs != std_n_recvs)
    {
        fprintf(stderr, "Incorrect NRecvs! New %d, Std %d\n", n_recvs, std_n_recvs);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    if (s_recvs != std_s_recvs)
    {
        fprintf(stderr, "Incorrect SRecvs! New %d, Std %d\n", s_recvs, std_s_recvs);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    for (int i = 0; i < n_recvs; i++)
    {
        proc = src[i];
        if (recvcounts[i] != proc_counts[proc])
        {
            fprintf(stderr, "Recv %d, Recvcounts %d, ProcCounts[%d] %d\n", 
                    i, recvcounts[i], proc, proc_counts[proc]);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        for (int j = 0; j < recvcounts[i]; j++)
        {
            if (recvvals[rdispls[i] + j] - first_col != indices[proc_displs[proc] + j])
            {
                fprintf(stderr, "Recv %d, Position %d, Recvvals %d, Indices %d\n", 
                        i, j, recvvals[rdispls[i] + j] - first_col, indices[proc_displs[proc] + j]);
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
        }
    }
}
