#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>

#include "neighbor_data.hpp"

void compare_neighbor_alltoallv_results(std::vector<int>& pmpi_recv_vals, std::vector<int>& mpix_recv_vals, int s)
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

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

// Get MPI Information
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Initial communication info (standard)
    int local_size = 10000; // Number of variables each rank stores
    MPIX_Data<int> send_data;
    MPIX_Data<int> recv_data;
    form_initial_communicator(local_size, &send_data, &recv_data);
    std::vector<long> global_send_idx(send_data.size_msgs);
    std::vector<long> global_recv_idx(recv_data.size_msgs);
    form_global_indices(local_size, send_data, recv_data, global_send_idx, global_recv_idx);

    std::vector<int> pmpi_recv_vals(recv_data.size_msgs);
    std::vector<int> mpix_recv_vals(recv_data.size_msgs);

    std::vector<int> send_vals(local_size);
    int val = local_size*rank;
    for (int i = 0; i < local_size; i++)
    {
        send_vals[i] = val++;
    }
    std::vector<int> alltoallv_send_vals(send_data.size_msgs);
    for (int i = 0; i < send_data.size_msgs; i++)
        alltoallv_send_vals[i] = send_vals[send_data.indices[i]];

    // Some MPI versions require sendcounts and recvcounts
    // to be non-NULL
    int* send_counts = send_data.counts.data();
    if (send_data.counts.data() == NULL)
        send_counts = new int[1];
    int* recv_counts = recv_data.counts.data();
    if (recv_data.counts.data() == NULL)
        recv_counts = new int[1];


    MPI_Comm std_comm;
    MPI_Status status;
    MPIX_Comm* xcomm;
    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);
    update_locality(xcomm, 4);
    MPIX_Request* xrequest;

    MPIX_Info* xinfo;
    MPIX_Info_init(&xinfo);

    MPIX_Topo* topo;
    MPIX_Topo_init(recv_data.num_msgs,
            recv_data.procs.data(), 
            recv_data.counts.data(),
            send_data.num_msgs, 
            send_data.procs.data(),
            send_data.counts.data(),
            xinfo, 
            &topo);

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

    // Standard MPI Implementation of Alltoallv
    MPI_Neighbor_alltoallv(alltoallv_send_vals.data(), 
            send_counts,
            send_data.indptr.data(), 
            MPI_INT,
            pmpi_recv_vals.data(), 
            recv_counts,
            recv_data.indptr.data(), 
            MPI_INT,
            std_comm);


    // Simple Persistent MPI Advance Implementation
    MPIX_Neighbor_topo_alltoallv_init(alltoallv_send_vals.data(), 
            send_data.counts.data(),
            send_data.indptr.data(), 
            MPI_INT,
            mpix_recv_vals.data(), 
            recv_data.counts.data(),
            recv_data.indptr.data(), 
            MPI_INT,
            topo,
            xcomm, 
            xinfo,
            &xrequest);
    MPIX_Start(xrequest);
    MPIX_Wait(xrequest, &status);
    MPIX_Request_free(&xrequest);
    compare_neighbor_alltoallv_results(pmpi_recv_vals, mpix_recv_vals, recv_data.size_msgs);
    

    // Locality-Aware MPI Advance Implementation
    std::fill(mpix_recv_vals.begin(), mpix_recv_vals.end(), 0);
    MPIX_Neighbor_locality_topo_alltoallv_init(alltoallv_send_vals.data(), 
            send_data.counts.data(),
            send_data.indptr.data(), 
            global_send_idx.data(),
            MPI_INT,
            mpix_recv_vals.data(), 
            recv_data.counts.data(),
            recv_data.indptr.data(), 
            global_recv_idx.data(),
            MPI_INT,
            topo,
            xcomm, 
            xinfo,
            &xrequest);
    MPIX_Start(xrequest);
    MPIX_Wait(xrequest, &status);
    MPIX_Request_free(&xrequest);
    compare_neighbor_alltoallv_results(pmpi_recv_vals, mpix_recv_vals, recv_data.size_msgs);


    // Partial Locality-Aware MPI Advance Implementation
    std::fill(mpix_recv_vals.begin(), mpix_recv_vals.end(), 0);
    MPIX_Neighbor_part_locality_topo_alltoallv_init(alltoallv_send_vals.data(), 
            send_data.counts.data(),
            send_data.indptr.data(), 
            MPI_INT,
            mpix_recv_vals.data(), 
            recv_data.counts.data(),
            recv_data.indptr.data(), 
            MPI_INT,
            topo,
            xcomm, 
            xinfo,
            &xrequest);
    MPIX_Start(xrequest);
    MPIX_Wait(xrequest, &status);
    MPIX_Request_free(&xrequest);
    compare_neighbor_alltoallv_results(pmpi_recv_vals, mpix_recv_vals, recv_data.size_msgs);

    MPIX_Topo_free(&topo);
    MPIX_Info_free(&xinfo);
    MPIX_Comm_free(&xcomm);
    MPI_Comm_free(&std_comm);

    if (send_data.counts.data() == NULL)
        delete[] send_counts;
    if (recv_data.counts.data() == NULL)
        delete[] recv_counts;

    MPI_Finalize();
    return 0;
} // end of main() //


