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

void compare_alltoallv_crs_results(int n_recvs, int send_msgs, int s_recvs, int send_size,
       int* src, std::vector<int>& proc_counts, int* recvcounts, std::vector<int>& proc_displs,
       std::vector<int>& send_idx, int* rdispls, long* recvvals, int first_col) 
{
    if (n_recvs != send_msgs)
    {
        fprintf(stderr, "NRecvs incorrect (%d), should be %d\n", n_recvs, send_msgs);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    if (s_recvs != send_size)
    {
        fprintf(stderr, "SRecvs incorrect (%d), should be %d\n", s_recvs, send_size);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    for (int i = 0; i < n_recvs; i++)
    {
        int proc = src[i];
        if (recvcounts[i] != proc_counts[proc])
        {
            fprintf(stderr, "Incorrect count at position %d, process %d, recvcounts %d, should be %d\n", 
                    i, proc, recvcounts[i], proc_counts[proc]);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        for (int j = 0; j < recvcounts[i]; j++)
        {
            if (recvvals[rdispls[i] + j] - first_col != send_idx[proc_displs[proc] + j])
            {
                fprintf(stderr, "Incorrect recvval from proc %d, position %d, getting %ld, should be %d\n", 
                        proc, j, recvvals[rdispls[i] + j] - first_col, send_idx[proc_displs[proc] + j]);
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
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
    MPIX_Comm_topo_init(xcomm);

    MPIX_Info* xinfo;
    MPIX_Info_init(&xinfo);

    // Update so there are 4 PPN rather than what MPI_Comm_split returns
    update_locality(xcomm, 4);

    // Read suitesparse matrix
    ParMat<int> A;
    readParMatrix(filename, A);
    form_comm(A);
    std::vector<int> proc_counts(num_procs, 0);
    std::vector<int> proc_displs(num_procs, 0);
    for (int i = 0; i < A.send_comm.n_msgs; i++)
    {
        int proc = A.send_comm.procs[i];
        proc_counts[proc] = A.send_comm.counts[i];
        proc_displs[proc] = A.send_comm.ptr[i];
    }

    int n_recvs, s_recvs, proc;
    int *src, *recvcounts, *rdispls;
    long *recvvals;

    /* TEST PERSONALIZED VERSION */
    s_recvs = -1;
    alltoallv_crs_personalized(A.recv_comm.n_msgs, A.recv_comm.size_msgs, A.recv_comm.procs.data(),
            A.recv_comm.counts.data(), A.recv_comm.ptr.data(), MPI_LONG,
            A.off_proc_columns.data(), 
            &n_recvs, &s_recvs, &src, &recvcounts, &rdispls, MPI_LONG, (void**)&recvvals, xinfo, xcomm);
    compare_alltoallv_crs_results(n_recvs, A.send_comm.n_msgs, s_recvs, A.send_comm.size_msgs, src,
            proc_counts, recvcounts, proc_displs, A.send_comm.idx, rdispls, recvvals, A.first_col);
    MPIX_Free(src);
    MPIX_Free(recvcounts);
    MPIX_Free(rdispls);
    MPIX_Free(recvvals);

    /* TEST NONBLOCKING VERSION */
    s_recvs = -1;
    alltoallv_crs_nonblocking(A.recv_comm.n_msgs, A.recv_comm.size_msgs, A.recv_comm.procs.data(),
            A.recv_comm.counts.data(), A.recv_comm.ptr.data(), MPI_LONG,
            A.off_proc_columns.data(), 
            &n_recvs, &s_recvs, &src, &recvcounts, &rdispls, MPI_LONG, (void**)&recvvals, xinfo, xcomm);
    compare_alltoallv_crs_results(n_recvs, A.send_comm.n_msgs, s_recvs, A.send_comm.size_msgs, src,
            proc_counts, recvcounts, proc_displs, A.send_comm.idx, rdispls, recvvals, A.first_col);
    MPIX_Free(src);
    MPIX_Free(recvcounts);
    MPIX_Free(rdispls);
    MPIX_Free(recvvals);

    /* TEST PERSONALIZED LOCALITY VERSION */
    s_recvs = -1;
    alltoallv_crs_personalized_loc(A.recv_comm.n_msgs, A.recv_comm.size_msgs, A.recv_comm.procs.data(),
            A.recv_comm.counts.data(), A.recv_comm.ptr.data(), MPI_LONG,
            A.off_proc_columns.data(), 
            &n_recvs, &s_recvs, &src, &recvcounts, &rdispls, MPI_LONG, (void**)&recvvals, xinfo, xcomm);
    compare_alltoallv_crs_results(n_recvs, A.send_comm.n_msgs, s_recvs, A.send_comm.size_msgs, src,
            proc_counts, recvcounts, proc_displs, A.send_comm.idx, rdispls, recvvals, A.first_col);
    MPIX_Free(src);
    MPIX_Free(recvcounts);
    MPIX_Free(rdispls);
    MPIX_Free(recvvals);

    /* TEST PERSONALIZED LOCALITY VERSION */
    s_recvs = -1;
    alltoallv_crs_nonblocking_loc(A.recv_comm.n_msgs, A.recv_comm.size_msgs, A.recv_comm.procs.data(),
            A.recv_comm.counts.data(), A.recv_comm.ptr.data(), MPI_LONG,
            A.off_proc_columns.data(), 
            &n_recvs, &s_recvs, &src, &recvcounts, &rdispls, MPI_LONG, (void**)&recvvals, xinfo, xcomm);
    compare_alltoallv_crs_results(n_recvs, A.send_comm.n_msgs, s_recvs, A.send_comm.size_msgs, src,
            proc_counts, recvcounts, proc_displs, A.send_comm.idx, rdispls, recvvals, A.first_col);
    MPIX_Free(src);
    MPIX_Free(recvcounts);
    MPIX_Free(rdispls);
    MPIX_Free(recvvals);
    
    MPIX_Info_free(&xinfo);
    MPIX_Comm_free(&xcomm);
}


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    test_matrix("../../../../test_data/dwt_162.pm");
    test_matrix("../../../../test_data/odepa400.pm");
    test_matrix("../../../../test_data/ww_36_pmec_36.pm");
    test_matrix("../../../../test_data/bcsstk01.pm");
    test_matrix("../../../../test_data/west0132.pm");
    test_matrix("../../../../test_data/gams10a.pm");
    test_matrix("../../../../test_data/gams10am.pm");
    test_matrix("../../../../test_data/D_10.pm");
    test_matrix("../../../../test_data/oscil_dcop_11.pm");
    test_matrix("../../../../test_data/tumorAntiAngiogenesis_4.pm");
    test_matrix("../../../../test_data/ch5-5-b1.pm");
    test_matrix("../../../../test_data/msc01050.pm");
    test_matrix("../../../../test_data/SmaGri.pm");
    test_matrix("../../../../test_data/radfr1.pm");
    test_matrix("../../../../test_data/bibd_49_3.pm");
    test_matrix("../../../../test_data/can_1054.pm");
    test_matrix("../../../../test_data/can_1072.pm");
    test_matrix("../../../../test_data/lp_woodw.pm");
    test_matrix("../../../../test_data/lp_sctap2.pm");
    MPI_Finalize();
    return 0;
} // end of main() //


