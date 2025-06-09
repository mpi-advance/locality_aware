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

void compare_neighbor_alltoallw_results(std::vector<int>& pmpi_recv_vals, std::vector<int>& mpix_recv_vals, int s)
{
    for (int i = 0; i < s; i++)
    {
        if (pmpi_recv_vals[i] != mpix_recv_vals[i])
        {
            fprintf(stderr, "PMPI recv != MPIX: position %d, pmpi %d, mpix %d\n", i, 
                    pmpi_recv_vals[i], mpix_recv_vals[i]);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void test_matrix(const char* filename)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_rank(MPI_COMM_WORLD, &num_procs);

    // Read suitesparse matrix
    ParMat<MPI_Aint> A;
    readParMatrix(filename, A);
    form_comm(A);

    std::vector<int> send_vals(A.on_proc.n_rows);
    std::iota(send_vals.begin(), send_vals.end(), 0);
    std::vector<int> alltoallv_send_vals(A.send_comm.size_msgs);
    for (int i = 0; i < A.send_comm.size_msgs; i++)
        alltoallv_send_vals[i] = send_vals[A.send_comm.idx[i]];

    std::vector<int> pmpi_recv_vals(A.recv_comm.size_msgs);
    std::vector<int> mpix_recv_vals(A.recv_comm.size_msgs);

    communicate(A, send_vals, mpix_recv_vals, MPI_INT);

    int int_size = sizeof(int);
    for (int i = 0; i < A.send_comm.n_msgs; i++)
        A.send_comm.ptr[i+1] *= int_size;
    for (int i = 0; i < A.recv_comm.n_msgs; i++)
        A.recv_comm.ptr[i+1] *= int_size;
    
    int* send_counts = A.send_comm.counts.data();
    if (A.send_comm.counts.data() == NULL)
        send_counts = new int[1];
    int* recv_counts = A.recv_comm.counts.data();
    if (A.recv_comm.counts.data() == NULL)
        recv_counts = new int[1];


    std::vector<MPI_Datatype> sendtypes(A.send_comm.n_msgs, MPI_INT);
    std::vector<MPI_Datatype> recvtypes(A.recv_comm.n_msgs, MPI_INT);

    MPI_Status status;
    MPI_Comm std_comm;

    MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD,
            A.recv_comm.n_msgs,
            A.recv_comm.procs.data(), 
            A.recv_comm.counts.data(),
            A.send_comm.n_msgs, 
            A.send_comm.procs.data(),
            A.send_comm.counts.data(),
            MPI_INFO_NULL, 
            0, 
            &std_comm);
    MPI_Neighbor_alltoallw(alltoallv_send_vals.data(), 
            send_counts,
            A.send_comm.ptr.data(), 
            sendtypes.data(),
            pmpi_recv_vals.data(), 
            recv_counts,
            A.recv_comm.ptr.data(), 
            recvtypes.data(),
            std_comm);
    MPI_Comm_free(&std_comm);
    //compare_neighbor_alltoallw_results(pmpi_recv_vals, mpix_recv_vals, A.recv_comm.size_msgs);


    // 2. Node-Aware Communication
    MPIX_Info* xinfo;
    MPIX_Info_init(&xinfo);

    MPIX_Comm* xcomm;
    MPIX_Request* xrequest;
    MPIX_Dist_graph_create_adjacent(MPI_COMM_WORLD,
            A.recv_comm.n_msgs,
            A.recv_comm.procs.data(), 
            A.recv_comm.counts.data(),
            A.send_comm.n_msgs, 
            A.send_comm.procs.data(),
            A.send_comm.counts.data(),
            MPI_INFO_NULL, 
            0, 
            &xcomm);
    update_locality(xcomm, 4);

    std::fill(mpix_recv_vals.begin(), mpix_recv_vals.end(), 0);
    MPIX_Neighbor_alltoallw_init(alltoallv_send_vals.data(), 
            A.send_comm.counts.data(),
            A.send_comm.ptr.data(), 
            sendtypes.data(),
            mpix_recv_vals.data(), 
            A.recv_comm.counts.data(),
            A.recv_comm.ptr.data(), 
            recvtypes.data(),
            xcomm, 
            xinfo,
            &xrequest);

    MPIX_Start(xrequest);
    MPIX_Wait(xrequest, &status);
    //compare_neighbor_alltoallw_results(pmpi_recv_vals, mpix_recv_vals, A.recv_comm.size_msgs);
    
    MPIX_Info_free(&xinfo);
    MPIX_Request_free(&xrequest);
    MPIX_Comm_free(&xcomm);

    if (A.send_comm.counts.data() == NULL)
        delete[] send_counts;
    if (A.recv_comm.counts.data() == NULL)
        delete[] recv_counts;

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


