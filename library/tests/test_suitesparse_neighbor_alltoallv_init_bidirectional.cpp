#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>

#include <iostream>
#include <numeric>
#include <set>
#include <vector>

#include "locality_aware.h"
#include "tests/common.hpp"
#include "tests/par_binary_IO.hpp"
#include "tests/sparse_mat.hpp"

void compare_neighbor_alltoallv_results(std::vector<int>& pmpi_recv_vals,
                                        std::vector<int>& mpil_recv_vals,
                                        int s)
{
    for (int i = 0; i < s; i++)
    {
        if (pmpi_recv_vals[i] != mpil_recv_vals[i])
        {
            fprintf(stderr,
                    "PMPI recv != MPIL: position %d, pmpi %d, mpil %d\n",
                    i,
                    pmpi_recv_vals[i],
                    mpil_recv_vals[i]);
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
    int idx, ctr, proc, size;
    readParMatrix(filename, A);
    form_comm(A);

    // Create list of all processes
    std::vector<int> procs(num_procs);
    std::iota(procs.begin(), procs.end(), 0);
    std::vector<int> proc_send_pos(num_procs);
    std::vector<int> proc_recv_pos(num_procs);

    std::vector<int> proc_send_sizes(num_procs, 0);
    std::vector<int> proc_recv_sizes(num_procs, 0);
    std::vector<int> proc_send_displs(num_procs + 1);
    std::vector<int> proc_recv_displs(num_procs + 1);
    std::vector<long> proc_send_indices(A.send_comm.size_msgs);
    std::vector<long> proc_recv_indices(A.recv_comm.size_msgs);

    std::vector<int> pmpi_recv_vals, mpil_recv_vals;
    std::vector<int> send_vals, alltoallv_send_vals;

    if (A.on_proc.n_cols)
    {
        send_vals.resize(A.on_proc.n_cols);
        std::iota(send_vals.begin(), send_vals.end(), 0);
        for (int i = 0; i < A.on_proc.n_cols; i++)
        {
            send_vals[i] += (rank * 1000);
        }
    }

    if (A.recv_comm.size_msgs)
    {
        pmpi_recv_vals.resize(A.recv_comm.size_msgs);
        mpil_recv_vals.resize(A.recv_comm.size_msgs);
    }

    if (A.send_comm.size_msgs)
    {
        alltoallv_send_vals.resize(A.send_comm.size_msgs);
    }

    // Create dense communication graph
    for (int i = 0; i < A.send_comm.n_msgs; i++)
    {
        proc                  = A.send_comm.procs[i];
        size                  = A.send_comm.counts[i];
        proc_send_sizes[proc] = size;
        proc_send_pos[proc]   = i;
    }
    for (int i = 0; i < A.recv_comm.n_msgs; i++)
    {
        proc                  = A.recv_comm.procs[i];
        size                  = A.recv_comm.counts[i];
        proc_recv_sizes[proc] = size;
        proc_recv_pos[proc]   = i;
    }
    proc_send_displs[0] = 0;
    proc_recv_displs[0] = 0;
    for (int i = 0; i < num_procs; i++)
    {
        ctr = proc_send_displs[i];
        if (proc_send_sizes[i])
        {
            idx = proc_send_pos[i];
            for (int j = A.send_comm.ptr[idx]; j < A.send_comm.ptr[idx + 1]; j++)
            {
                alltoallv_send_vals[ctr] = send_vals[A.send_comm.idx[j]];
                proc_send_indices[ctr++] = A.send_comm.idx[j] + A.first_col;
            }
        }
        proc_send_displs[i + 1] = ctr;

        ctr = proc_recv_displs[i];
        if (proc_recv_sizes[i])
        {
            idx = proc_recv_pos[i];
            for (int j = A.recv_comm.ptr[idx]; j < A.recv_comm.ptr[idx + 1]; j++)
            {
                proc_recv_indices[ctr++] = A.off_proc_columns[j];
            }
        }
        proc_recv_displs[i + 1] = ctr;
    }

    // MPI and MPIL Variables
    MPI_Status status;
    MPIL_Comm* xcomm;
    MPIL_Request* xrequest;
    MPIL_Info* xinfo;
    MPIL_Info_init(&xinfo);

    // Create standard PMPI neighbor communicator
    MPI_Comm std_comm;
    PMPI_Dist_graph_create_adjacent(MPI_COMM_WORLD,
                                    num_procs,
                                    procs.data(),
                                    MPI_UNWEIGHTED,
                                    num_procs,
                                    procs.data(),
                                    MPI_UNWEIGHTED,
                                    MPI_INFO_NULL,
                                    0,
                                    &std_comm);

    // Standard PMPI neighbor exchange
    PMPI_Neighbor_alltoallv(alltoallv_send_vals.data(),
                            proc_send_sizes.data(),
                            proc_send_displs.data(),
                            MPI_INT,
                            pmpi_recv_vals.data(),
                            proc_recv_sizes.data(),
                            proc_recv_displs.data(),
                            MPI_INT,
                            std_comm);

    PMPI_Comm_free(&std_comm);

    // MPI Advance neighbor communicator
    MPIL_Dist_graph_create_adjacent(MPI_COMM_WORLD,
                                    num_procs,
                                    procs.data(),
                                    MPI_UNWEIGHTED,
                                    num_procs,
                                    procs.data(),
                                    MPI_UNWEIGHTED,
                                    xinfo,
                                    0,
                                    &xcomm);
    MPIL_Comm_update_locality(xcomm, 4);

    // Standard exchange
    mpil_neighbor_alltoallv_implementation = NEIGHBOR_ALLTOALLV_STANDARD;
    std::fill(mpil_recv_vals.begin(), mpil_recv_vals.end(), 0);
    MPIL_Neighbor_alltoallv(alltoallv_send_vals.data(),
                            proc_send_sizes.data(),
                            proc_send_displs.data(),
                            MPI_INT,
                            mpil_recv_vals.data(),
                            proc_recv_sizes.data(),
                            proc_recv_displs.data(),
                            MPI_INT,
                            xcomm);
    compare_neighbor_alltoallv_results(
        pmpi_recv_vals, mpil_recv_vals, A.recv_comm.size_msgs);

    // 2. Node-Aware Communication
    mpil_neighbor_alltoallv_init_implementation = NEIGHBOR_ALLTOALLV_INIT_STANDARD;
    std::fill(mpil_recv_vals.begin(), mpil_recv_vals.end(), 0);
    MPIL_Neighbor_alltoallv_init(alltoallv_send_vals.data(),
                                 proc_send_sizes.data(),
                                 proc_send_displs.data(),
                                 MPI_INT,
                                 mpil_recv_vals.data(),
                                 proc_recv_sizes.data(),
                                 proc_recv_displs.data(),
                                 MPI_INT,
                                 xcomm,
                                 xinfo,
                                 &xrequest);

    MPIL_Start(xrequest);
    MPIL_Wait(xrequest, &status);
    MPIL_Request_free(&xrequest);
    compare_neighbor_alltoallv_results(
        pmpi_recv_vals, mpil_recv_vals, A.recv_comm.size_msgs);

    // 3. MPI Advance - Optimized Communication
    mpil_neighbor_alltoallv_init_implementation = NEIGHBOR_ALLTOALLV_INIT_LOCALITY;
    std::fill(mpil_recv_vals.begin(), mpil_recv_vals.end(), 0);
    MPIL_Neighbor_alltoallv_init(alltoallv_send_vals.data(),
                                 proc_send_sizes.data(),
                                 proc_send_displs.data(),
                                 MPI_INT,
                                 mpil_recv_vals.data(),
                                 proc_recv_sizes.data(),
                                 proc_recv_displs.data(),
                                 MPI_INT,
                                 xcomm,
                                 xinfo,
                                 &xrequest);

    MPIL_Start(xrequest);
    MPIL_Wait(xrequest, &status);
    MPIL_Request_free(&xrequest);
    compare_neighbor_alltoallv_results(
        pmpi_recv_vals, mpil_recv_vals, A.recv_comm.size_msgs);

    // Standard from Extended Interface
    mpil_neighbor_alltoallv_init_implementation = NEIGHBOR_ALLTOALLV_INIT_STANDARD;
    std::fill(mpil_recv_vals.begin(), mpil_recv_vals.end(), 0);
    MPIL_Neighbor_alltoallv_init_ext(alltoallv_send_vals.data(),
                                     proc_send_sizes.data(),
                                     proc_send_displs.data(),
                                     proc_send_indices.data(),
                                     MPI_INT,
                                     mpil_recv_vals.data(),
                                     proc_recv_sizes.data(),
                                     proc_recv_displs.data(),
                                     proc_recv_indices.data(),
                                     MPI_INT,
                                     xcomm,
                                     xinfo,
                                     &xrequest);
    MPIL_Start(xrequest);
    MPIL_Wait(xrequest, &status);
    MPIL_Request_free(&xrequest);
    compare_neighbor_alltoallv_results(
        pmpi_recv_vals, mpil_recv_vals, A.recv_comm.size_msgs);

    // Full Locality
    mpil_neighbor_alltoallv_init_implementation = NEIGHBOR_ALLTOALLV_INIT_LOCALITY;
    std::fill(mpil_recv_vals.begin(), mpil_recv_vals.end(), 0);
    MPIL_Neighbor_alltoallv_init_ext(alltoallv_send_vals.data(),
                                     proc_send_sizes.data(),
                                     proc_send_displs.data(),
                                     proc_send_indices.data(),
                                     MPI_INT,
                                     mpil_recv_vals.data(),
                                     proc_recv_sizes.data(),
                                     proc_recv_displs.data(),
                                     proc_recv_indices.data(),
                                     MPI_INT,
                                     xcomm,
                                     xinfo,
                                     &xrequest);
    MPIL_Start(xrequest);
    MPIL_Wait(xrequest, &status);
    MPIL_Request_free(&xrequest);
    compare_neighbor_alltoallv_results(
        pmpi_recv_vals, mpil_recv_vals, A.recv_comm.size_msgs);

    MPIL_Info_free(&xinfo);
    MPIL_Comm_free(&xcomm);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    test_all_matrices();

    MPI_Finalize();
    return 0;
}  // end of main()