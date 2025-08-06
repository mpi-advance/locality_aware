#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <numeric>
#include <set>

#include "tests/sparse_mat.hpp"
#include "tests/par_binary_IO.hpp"

void compare_alltoallv_results(std::vector<int>& pmpi, std::vector<int>& mpix, int s)
{
    for (int i = 0; i < s; i++)
    {
        if (pmpi[i] != mpix[i])
        {
            fprintf(stderr, "MPIX Alltoallv != PMPI, position %d, pmpi %d, mpix %d\n", 
                    i, pmpi[i], mpix[i]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

}

void test_matrix(const char* filename)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    MPIX_Comm* xcomm;
    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);
    update_locality(xcomm, 4);

    // Read suitesparse matrix
    ParMat<int> A;
    readParMatrix(filename, A);
    form_comm(A);

    std::vector<int> send_vals(A.on_proc.n_rows);
    std::iota(send_vals.begin(), send_vals.end(), 0);
    for (int i = 0; i < A.on_proc.n_rows; i++)
        send_vals[i] += (rank*1000);

    // Alltoallv_send_vals must be ordered (dest 0 to num_procs-1)
    std::vector<int> proc_pos(num_procs, -1);
    for (int i = 0; i < A.send_comm.n_msgs; i++)
        proc_pos[A.send_comm.procs[i]] = i;

    std::vector<int> alltoallv_send_vals(A.send_comm.size_msgs);
    int start, end, idx;
    int ctr = 0;
    for (int i = 0; i < num_procs; i++)
    {
        idx = proc_pos[i];
        if (proc_pos[i] < 0) continue;

        start = A.send_comm.ptr[idx];
        end = A.send_comm.ptr[idx+1];
        for (int j = start; j < end; j++)
        {
            alltoallv_send_vals[ctr++] = send_vals[A.send_comm.idx[j]];
        }
    }

    std::vector<int> sendcounts(num_procs, 0); 
    std::vector<int> sdispls(num_procs+1);
    std::vector<int> recvcounts(num_procs, 0);
    std::vector<int> rdispls(num_procs+1);

    for (int i = 0; i < A.send_comm.n_msgs; i++)
        sendcounts[A.send_comm.procs[i]] = A.send_comm.ptr[i+1] - A.send_comm.ptr[i];
    for (int i = 0; i < A.recv_comm.n_msgs; i++)
        recvcounts[A.recv_comm.procs[i]] = A.recv_comm.ptr[i+1] - A.recv_comm.ptr[i];

    sdispls[0] = 0;
    rdispls[0] = 0;
    for (int i = 0; i < num_procs; i++)
    {
        sdispls[i+1] = sdispls[i] + sendcounts[i];
        rdispls[i+1] = rdispls[i] + recvcounts[i];
    }

    std::vector<int> pmpi_recv_vals(A.recv_comm.size_msgs);
    std::vector<int> mpix_recv_vals(A.recv_comm.size_msgs);

    communicate(A, send_vals, mpix_recv_vals, MPI_INT);

    PMPI_Alltoallv(alltoallv_send_vals.data(), 
            sendcounts.data(),
            sdispls.data(),
            MPI_INT,
            pmpi_recv_vals.data(),
            recvcounts.data(),
            rdispls.data(),
            MPI_INT,
            MPI_COMM_WORLD);
    compare_alltoallv_results(pmpi_recv_vals, mpix_recv_vals, A.recv_comm.size_msgs);

    mpix_alltoallv_implementation = ALLTOALLV_PMPI;
    std::fill(mpix_recv_vals.begin(), mpix_recv_vals.end(), 0);
    MPIX_Alltoallv(alltoallv_send_vals.data(), 
            sendcounts.data(),
            sdispls.data(),
            MPI_INT,
            mpix_recv_vals.data(),
            recvcounts.data(),
            rdispls.data(),
            MPI_INT,
            xcomm);
    compare_alltoallv_results(pmpi_recv_vals, mpix_recv_vals, A.recv_comm.size_msgs);

    mpix_alltoallv_implementation = ALLTOALLV_PAIRWISE;
    std::fill(mpix_recv_vals.begin(), mpix_recv_vals.end(), 0);
    MPIX_Alltoallv(alltoallv_send_vals.data(), 
            sendcounts.data(),
            sdispls.data(),
            MPI_INT,
            mpix_recv_vals.data(),
            recvcounts.data(),
            rdispls.data(),
            MPI_INT,
            xcomm);
    compare_alltoallv_results(pmpi_recv_vals, mpix_recv_vals, A.recv_comm.size_msgs);
    
    mpix_alltoallv_implementation = ALLTOALLV_NONBLOCKING;
    std::fill(mpix_recv_vals.begin(), mpix_recv_vals.end(), 0);
    MPIX_Alltoallv(alltoallv_send_vals.data(), 
            sendcounts.data(),
            sdispls.data(),
            MPI_INT,
            mpix_recv_vals.data(),
            recvcounts.data(),
            rdispls.data(),
            MPI_INT,
            xcomm);
    compare_alltoallv_results(pmpi_recv_vals, mpix_recv_vals, A.recv_comm.size_msgs);

    mpix_alltoallv_implementation = ALLTOALLV_BATCH;
    std::fill(mpix_recv_vals.begin(), mpix_recv_vals.end(), 0);
    MPIX_Alltoallv(alltoallv_send_vals.data(), 
            sendcounts.data(),
            sdispls.data(),
            MPI_INT,
            mpix_recv_vals.data(),
            recvcounts.data(),
            rdispls.data(),
            MPI_INT,
            xcomm);
    compare_alltoallv_results(pmpi_recv_vals, mpix_recv_vals, A.recv_comm.size_msgs);

    mpix_alltoallv_implementation = ALLTOALLV_BATCH_ASYNC;
    std::fill(mpix_recv_vals.begin(), mpix_recv_vals.end(), 0);
    MPIX_Alltoallv(alltoallv_send_vals.data(), 
            sendcounts.data(),
            sdispls.data(),
            MPI_INT,
            mpix_recv_vals.data(),
            recvcounts.data(),
            rdispls.data(),
            MPI_INT,
            xcomm);
    compare_alltoallv_results(pmpi_recv_vals, mpix_recv_vals, A.recv_comm.size_msgs);

    MPIX_Comm_free(&xcomm);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    test_matrix("../../../../test_data/dwt_162.pm");
    test_matrix("../../../../test_data/odepa400.pm");
    test_matrix("../../../../test_data/ww_36_pmec_36.pm");

    MPI_Finalize();
    return 0;
} // end of main() //


