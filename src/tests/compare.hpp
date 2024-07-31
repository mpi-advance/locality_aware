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
    ASSERT_EQ(n_recvs, std_n_recvs);
    for (int i = 0; i < n_recvs; i++)
        ASSERT_EQ(recvvals[i], std_recvvals[src[i]]);
}

void compare(int n_recvs, int std_n_recvs, int s_recvs, int std_s_recvs,
    std::vector<int>& src, std::vector<int>& recvcounts,
    std::vector<int>& rdispls, std::vector<long>& recvvals,
    int first_col, std::vector<int>& proc_counts,
    std::vector<int>& proc_displs, std::vector<int>& indices)
{
    int proc;
    ASSERT_EQ(n_recvs, std_n_recvs);
    ASSERT_EQ(s_recvs, std_s_recvs);
    for (int i = 0; i < n_recvs; i++)
    {
        proc = src[i];
        ASSERT_EQ(recvcounts[i], proc_counts[proc]);
        for (int j = 0; j < recvcounts[i]; j++)
            ASSERT_EQ(recvvals[rdispls[i] + j] - first_col,
                    indices[proc_displs[proc] + j]);
    }
}
