#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>

#include <iostream>
#include <set>
#include <vector>

//#include "persistent/MPIL_Request.h"
#include "locality_aware.h"
#include "neighbor_data.hpp"

void compare_neighbor_alltoallv_results(std::vector<int>& pmpi_recv_vals,
                                        std::vector<int>& mpix_recv_vals,
                                        int s)
{
    for (int i = 0; i < s; i++)
    {
        if (pmpi_recv_vals[i] != mpix_recv_vals[i])
        {
            fprintf(stderr,
                    "PMPI recv != MPIL: position %d, pmpi %d, mpix %d\n",
                    i,
                    pmpi_recv_vals[i],
                    mpix_recv_vals[i]);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    // Get MPI Information
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Initial communication info (standard)
    int local_size = 10000;  // Number of variables each rank stores
    MPIL_Data<int> send_data;
    MPIL_Data<int> recv_data;
    form_initial_communicator(local_size, &send_data, &recv_data);
    std::vector<long> global_send_idx(send_data.size_msgs);
    std::vector<long> global_recv_idx(recv_data.size_msgs);
    form_global_indices(
        local_size, send_data, recv_data, global_send_idx, global_recv_idx);

    // Test correctness of communication
    std::vector<int> mpix_recv_vals(recv_data.size_msgs);
    std::vector<int> pmpi_recv_vals(recv_data.size_msgs);

    std::vector<int> send_vals(local_size);
    int val = local_size * rank;
    for (int i = 0; i < local_size; i++)
    {
        send_vals[i] = val++;
    }

    std::vector<int> alltoallv_send_vals(send_data.size_msgs);
    for (int i = 0; i < send_data.size_msgs; i++)
    {
        alltoallv_send_vals[i] = send_vals[send_data.indices[i]];
    }

    MPI_Comm std_comm;
    MPI_Status status;
    MPIL_Comm* xcomm;
    MPIL_Request* xrequest;

    MPIL_Info* xinfo;
    MPIL_Info_init(&xinfo);

    // Standard MPI Dist Graph Create
    MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD,
                                   recv_data.num_msgs,
                                   recv_data.procs.data(),
                                   recv_data.counts.data(),
                                   send_data.num_msgs,
                                   send_data.procs.data(),
                                   send_data.counts.data(),
                                   MPI_INFO_NULL,
                                   0,
                                   &std_comm);

    // MPI Advance Dist Graph Create
    MPIL_Dist_graph_create_adjacent(MPI_COMM_WORLD,
                                    recv_data.num_msgs,
                                    recv_data.procs.data(),
                                    recv_data.counts.data(),
                                    send_data.num_msgs,
                                    send_data.procs.data(),
                                    send_data.counts.data(),
                                    xinfo,
                                    0,
                                    &xcomm);

    // Update Locality : 4 PPN (for single-node tests)
    MPIL_Comm_update_locality(xcomm, 4);

    // Standard MPI Implementation of Alltoallv
    int* send_counts = send_data.counts.data();
    if (send_data.counts.data() == NULL)
    {
        send_counts = new int[1];
    }
    int* recv_counts = recv_data.counts.data();
    if (recv_data.counts.data() == NULL)
    {
        recv_counts = new int[1];
    }
    MPI_Neighbor_alltoallv(alltoallv_send_vals.data(),
                           send_counts,
                           send_data.indptr.data(),
                           MPI_INT,
                           pmpi_recv_vals.data(),
                           recv_counts,
                           recv_data.indptr.data(),
                           MPI_INT,
                           std_comm);
    if (send_data.counts.data() == NULL)
    {
        delete[] send_counts;
    }
    if (recv_data.counts.data() == NULL)
    {
        delete[] recv_counts;
    }

    // Simple Persistent MPI Advance Implementation
    MPIL_Neighbor_alltoallv_init(alltoallv_send_vals.data(),
                                 send_data.counts.data(),
                                 send_data.indptr.data(),
                                 MPI_INT,
                                 mpix_recv_vals.data(),
                                 recv_data.counts.data(),
                                 recv_data.indptr.data(),
                                 MPI_INT,
                                 xcomm,
                                 xinfo,
                                 &xrequest);

    // Reorder during first send/recv
    std::fill(mpix_recv_vals.begin(), mpix_recv_vals.end(), 0);
    MPIL_Request_reorder(xrequest, 1);
    MPIL_Start(xrequest);
    MPIL_Wait(xrequest, &status);
    compare_neighbor_alltoallv_results(
        pmpi_recv_vals, mpix_recv_vals, recv_data.size_msgs);

    // Standard send/recv with reordered recvs
    std::fill(mpix_recv_vals.begin(), mpix_recv_vals.end(), 0);
    MPIL_Start(xrequest);
    MPIL_Wait(xrequest, &status);
    compare_neighbor_alltoallv_results(
        pmpi_recv_vals, mpix_recv_vals, recv_data.size_msgs);

    MPIL_Request_free(&xrequest);
    MPIL_Info_free(&xinfo);
    MPIL_Comm_free(&xcomm);
    MPI_Comm_free(&std_comm);

    MPI_Finalize();
    return 0;
}  // end of main() //
