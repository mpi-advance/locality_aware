#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>

#include "neighbor_data.hpp"


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

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

 // Get MPI Information
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Initial communication info (standard)
    int local_size = 10000; // Number of variables each rank stores
    MPIX_Data<MPI_Aint> send_data;
    MPIX_Data<MPI_Aint> recv_data;
    form_initial_communicator(local_size, &send_data, &recv_data);
    int int_size = sizeof(int);
    for (int i = 0; i < send_data.num_msgs; i++)
        send_data.indptr[i+1] *= int_size;
    for (int i = 0; i < recv_data.num_msgs; i++)
        recv_data.indptr[i+1] *= int_size;

    // Test correctness of communication
    std::vector<int> send_vals(local_size);
    int val = local_size*rank;
    for (int i = 0; i < local_size; i++)
    {
        send_vals[i] = val++;
    }
    std::vector<int> alltoallv_send_vals(send_data.size_msgs);
    for (int i = 0; i < send_data.size_msgs; i++)
        alltoallv_send_vals[i] = send_vals[send_data.indices[i]];
    
    // Required to be non-NULL for some version of MPI
    int* send_counts = send_data.counts.data();
    if (send_data.counts.data() == NULL)
        send_counts = new int[1];
    int* recv_counts = recv_data.counts.data();
    if (recv_data.counts.data() == NULL)
        recv_counts = new int[1];


    std::vector<int> pmpi_recv_vals(recv_data.size_msgs);
    std::vector<int> mpix_recv_vals(recv_data.size_msgs);


    MPI_Comm std_comm;
    MPI_Status status;
    MPIX_Comm* xcomm;
    MPIX_Request* xrequest;
    MPIX_Info* xinfo;
    std::vector<MPI_Datatype> sendtypes(num_procs, MPI_INT);
    std::vector<MPI_Datatype> recvtypes(num_procs, MPI_INT);


    PMPI_Dist_graph_create_adjacent(MPI_COMM_WORLD,
            recv_data.num_msgs, 
            recv_data.procs.data(), 
            recv_data.counts.data(),
            send_data.num_msgs, 
            send_data.procs.data(),
            send_data.counts.data(),
            MPI_INFO_NULL, 
            0, 
            &std_comm);

    MPI_Neighbor_alltoallw(alltoallv_send_vals.data(), 
            send_counts,
            send_data.indptr.data(), 
            sendtypes.data(),
            pmpi_recv_vals.data(), 
            recv_counts,
            recv_data.indptr.data(), 
            recvtypes.data(),
            std_comm);


    // 2. Node-Aware Communication
    MPIX_Dist_graph_create_adjacent(MPI_COMM_WORLD,
            recv_data.num_msgs, 
            recv_data.procs.data(), 
            recv_data.counts.data(),
            send_data.num_msgs, 
            send_data.procs.data(),
            send_data.counts.data(),
            MPI_INFO_NULL, 
            0, 
            &xcomm);

    MPIX_Info_init(&xinfo);
    MPIX_Neighbor_alltoallw_init(alltoallv_send_vals.data(), 
            send_data.counts.data(),
            send_data.indptr.data(), 
            sendtypes.data(),
            mpix_recv_vals.data(), 
            recv_data.counts.data(),
            recv_data.indptr.data(), 
            recvtypes.data(),
            xcomm, 
            xinfo,
            &xrequest);

    MPIX_Start(xrequest);
    MPIX_Wait(xrequest, &status);
    compare_neighbor_alltoallw_results(pmpi_recv_vals, mpix_recv_vals, recv_data.size_msgs);

    // Delete temp send/recv counts variables
    // That were needed because some versions of 
    // MPI require non-NULL counts array
    if (send_data.counts.data() == NULL)
        delete[] send_counts;
    if (recv_data.counts.data() == NULL)
        delete[] recv_counts;


    MPIX_Info_free(&xinfo);
    MPIX_Request_free(&xrequest);
    MPIX_Comm_free(&xcomm);
    MPI_Comm_free(&std_comm);


    MPI_Finalize();
    return 0;
} // end of main() //


