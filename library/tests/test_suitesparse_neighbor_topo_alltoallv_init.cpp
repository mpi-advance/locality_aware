#include "../../include/locality_aware.h"
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

void compare_neighbor_alltoallv_results(std::vector<int>& pmpi_recv_vals, std::vector<int>& mpix_recv_vals, int s)
{
    for (int i = 0; i < s; i++)
    {
        if (pmpi_recv_vals[i] != mpix_recv_vals[i])
        {
            fprintf(stderr, "PMPI recv != MPIL: position %d, pmpi %d, mpix %d\n", i, 
                    pmpi_recv_vals[i], mpix_recv_vals[i]);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void test_matrix(const char* filename)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Read suitesparse matrix
    ParMat<int> A;
    int idx;
    readParMatrix(filename, A);
    form_comm(A);

    std::vector<int> pmpi_recv_vals, mpix_recv_vals;
    std::vector<int> send_vals, alltoallv_send_vals;
    std::vector<long> send_indices;

    if (A.on_proc.n_cols)
    {
        send_vals.resize(A.on_proc.n_cols);
        std::iota(send_vals.begin(), send_vals.end(), 0);
        for (int i = 0; i < A.on_proc.n_cols; i++)
            send_vals[i] += (rank*1000);
    }

    if (A.recv_comm.size_msgs)
    {
        pmpi_recv_vals.resize(A.recv_comm.size_msgs);
        mpix_recv_vals.resize(A.recv_comm.size_msgs);
    }

    if (A.send_comm.size_msgs)
    {
        alltoallv_send_vals.resize(A.send_comm.size_msgs);
        send_indices.resize(A.send_comm.size_msgs);
        for (int i = 0; i < A.send_comm.size_msgs; i++)
        {
            idx = A.send_comm.idx[i];
            alltoallv_send_vals[i] = send_vals[idx];
            send_indices[i] = A.send_comm.idx[i] + A.first_col;
        }
    }

    communicate(A, send_vals, mpix_recv_vals, MPI_INT);

    MPI_Comm std_comm;
    MPI_Status status;

    MPIL_Request* xrequest;
    MPIL_Comm* xcomm;
    MPIL_Comm_init(&xcomm, MPI_COMM_WORLD);
    update_locality(xcomm, 4);
    MPIL_Info* xinfo;
    MPIL_Info_init(&xinfo);

    MPIL_Topo* topo;
    MPIL_Topo_init(A.recv_comm.n_msgs,
            A.recv_comm.procs.data(), 
            A.recv_comm.counts.data(),
            A.send_comm.n_msgs, 
            A.send_comm.procs.data(),
            A.send_comm.counts.data(),
            xinfo, 
            &topo);

    int* s = A.recv_comm.procs.data();
    if (A.recv_comm.n_msgs == 0)
        s = MPI_WEIGHTS_EMPTY;
    int* d = A.send_comm.procs.data();
    if (A.send_comm.n_msgs  == 0)
        d = MPI_WEIGHTS_EMPTY;

    PMPI_Dist_graph_create_adjacent(MPI_COMM_WORLD,
            A.recv_comm.n_msgs,
            s,
            MPI_UNWEIGHTED,
            A.send_comm.n_msgs, 
            d,
            MPI_UNWEIGHTED,
            MPI_INFO_NULL, 
            0, 
            &std_comm);

    int* send_counts = A.send_comm.counts.data();
    if (A.send_comm.counts.data() == NULL)
        send_counts = new int[1];
    int* recv_counts = A.recv_comm.counts.data();
    if (A.recv_comm.counts.data() == NULL)
        recv_counts = new int[1];
    PMPI_Neighbor_alltoallv(alltoallv_send_vals.data(),
            send_counts,
            A.send_comm.ptr.data(), 
            MPI_INT,
            pmpi_recv_vals.data(),
            recv_counts,
            A.recv_comm.ptr.data(),
            MPI_INT,
            std_comm);
    if (A.send_comm.counts.data() == NULL)
        delete[] send_counts;
    if (A.recv_comm.counts.data() == NULL)
        delete[] recv_counts;
    compare_neighbor_alltoallv_results(pmpi_recv_vals, mpix_recv_vals, A.recv_comm.size_msgs);

       
    mpix_neighbor_alltoallv_implementation = NEIGHBOR_ALLTOALLV_STANDARD;
    std::fill(mpix_recv_vals.begin(), mpix_recv_vals.end(), 0);
    MPIL_Neighbor_alltoallv_topo(alltoallv_send_vals.data(), 
            A.send_comm.counts.data(),
            A.send_comm.ptr.data(), 
            MPI_INT,
            mpix_recv_vals.data(), 
            A.recv_comm.counts.data(),
            A.recv_comm.ptr.data(), 
            MPI_INT,
            topo,
            xcomm);
    compare_neighbor_alltoallv_results(pmpi_recv_vals, mpix_recv_vals, A.recv_comm.size_msgs);


    mpix_neighbor_alltoallv_init_implementation = NEIGHBOR_ALLTOALLV_INIT_STANDARD;
    std::fill(mpix_recv_vals.begin(), mpix_recv_vals.end(), 0);
    MPIL_Neighbor_alltoallv_init_topo(alltoallv_send_vals.data(), 
            A.send_comm.counts.data(),
            A.send_comm.ptr.data(), 
            MPI_INT,
            mpix_recv_vals.data(), 
            A.recv_comm.counts.data(),
            A.recv_comm.ptr.data(), 
            MPI_INT,
            topo,
            xcomm, 
            xinfo,
            &xrequest);

    MPIL_Start(xrequest);
    MPIL_Wait(xrequest, &status);
    MPIL_Request_free(&xrequest);
    compare_neighbor_alltoallv_results(pmpi_recv_vals, mpix_recv_vals, A.recv_comm.size_msgs);


    mpix_neighbor_alltoallv_init_implementation = NEIGHBOR_ALLTOALLV_INIT_LOCALITY;
    std::fill(mpix_recv_vals.begin(), mpix_recv_vals.end(), 0);
    MPIL_Neighbor_alltoallv_init_topo(alltoallv_send_vals.data(), 
            A.send_comm.counts.data(),
            A.send_comm.ptr.data(), 
            MPI_INT,
            mpix_recv_vals.data(), 
            A.recv_comm.counts.data(),
            A.recv_comm.ptr.data(), 
            MPI_INT,
            topo,
            xcomm, 
            xinfo,
            &xrequest);

    MPIL_Start(xrequest);
    MPIL_Wait(xrequest, &status);
    MPIL_Request_free(&xrequest);
    compare_neighbor_alltoallv_results(pmpi_recv_vals, mpix_recv_vals, A.recv_comm.size_msgs);


    // 3. MPI Advance - Optimized Communication
    mpix_neighbor_alltoallv_init_implementation = NEIGHBOR_ALLTOALLV_INIT_STANDARD;
    std::fill(mpix_recv_vals.begin(), mpix_recv_vals.end(), 0);
    MPIL_Neighbor_alltoallv_init_ext_topo(alltoallv_send_vals.data(), 
            A.send_comm.counts.data(),
            A.send_comm.ptr.data(), 
            send_indices.data(),
            MPI_INT,
            mpix_recv_vals.data(), 
            A.recv_comm.counts.data(),
            A.recv_comm.ptr.data(), 
            A.off_proc_columns.data(),
            MPI_INT,
            topo,
            xcomm, 
            xinfo,
            &xrequest);

    MPIL_Start(xrequest);
    MPIL_Wait(xrequest, &status);
    MPIL_Request_free(&xrequest);
    compare_neighbor_alltoallv_results(pmpi_recv_vals, mpix_recv_vals, A.recv_comm.size_msgs);


    mpix_neighbor_alltoallv_init_implementation = NEIGHBOR_ALLTOALLV_INIT_LOCALITY;
    std::fill(mpix_recv_vals.begin(), mpix_recv_vals.end(), 0);
    MPIL_Neighbor_alltoallv_init_ext_topo(alltoallv_send_vals.data(), 
            A.send_comm.counts.data(),
            A.send_comm.ptr.data(), 
            send_indices.data(),
            MPI_INT,
            mpix_recv_vals.data(), 
            A.recv_comm.counts.data(),
            A.recv_comm.ptr.data(), 
            A.off_proc_columns.data(),
            MPI_INT,
            topo,
            xcomm, 
            xinfo,
            &xrequest);

    MPIL_Start(xrequest);
    MPIL_Wait(xrequest, &status);
    MPIL_Request_free(&xrequest);
    compare_neighbor_alltoallv_results(pmpi_recv_vals, mpix_recv_vals, A.recv_comm.size_msgs);

    MPIL_Topo_free(&topo);
    MPIL_Info_free(&xinfo);
    MPIL_Comm_free(&xcomm);
    PMPI_Comm_free(&std_comm);
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
    test_matrix("../../../../test_data/lp_sctap2.pm");
    test_matrix("../../../../test_data/lp_woodw.pm");
    MPI_Finalize();
    return 0;
} // end of main() //



