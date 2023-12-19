#include "mpi_advance.h"
#include "tests/sparse_mat.hpp"
#include "tests/par_binary_IO.hpp"

#include <numeric>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if(rank == 0)
	printf("Num ranks: %d\n", num_procs);

    double t0, tf;
    
    int iters = 100;

    if (argc == 1)
    {
        if (rank == 0) printf("Pass Matrix Filename as Command Line Arg!\n");
        MPI_Finalize();
        return 0;
    }
    char* filename = argv[1];

    // Read suitesparse matrix
    ParMat<int> A;
    readParMatrix(filename, A);

    // Form Communication Package (A.send_comm, A.recv_comm)
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    form_comm(A);
    tf = MPI_Wtime() - t0;
    MPI_Reduce(&tf, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) { printf("Standard comm form: %8e\n", t0); }

    // Each Process sends vals[i] = rank*1000 + i
    //    - For debugging, can tell where values originate
    //           and original index
    std::vector<int> send_vals;
    if (A.on_proc.n_cols)
    {
        send_vals.resize(A.on_proc.n_cols);
        for (int i = 0; i < A.on_proc.n_cols; i++)
            send_vals[i] = rank * 1000 + i;
    }


    // Allocate and Fill Packed Send Buffer
    std::vector<int> packed_send_vals;
    std::vector<int> pure_packed_send;
    if (A.send_comm.size_msgs)
    {
        packed_send_vals.resize(A.send_comm.size_msgs);
        pure_packed_send.resize(A.send_comm.size_msgs);
        for (int i = 0; i < A.send_comm.size_msgs; i++)
        {
            packed_send_vals[i] = send_vals[A.send_comm.idx[i]];
	    pure_packed_send[i] = send_vals[A.send_comm.idx[i]];
        }
    }

    // Allocate Recv Buffer
    std::vector<int> std_recv_buffer; // Standard Communication (for debugging comparison)
    std::vector<int> neigh_recv_buffer; // Neighbor Collective Communication (MPI Version)
    std::vector<int> mpix_recv_buffer; // Neighbor Collective (MPI Advance)
    if (A.recv_comm.size_msgs)
    {
        std_recv_buffer.resize(A.recv_comm.size_msgs);
        neigh_recv_buffer.resize(A.recv_comm.size_msgs);
        mpix_recv_buffer.resize(A.recv_comm.size_msgs);
    }

    // Standard Communication (for comparisons)
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < iters; i++){
    	communicate3(A, packed_send_vals, std_recv_buffer, MPI_INT);
    	communicate3_flip(A, packed_send_vals, std_recv_buffer, MPI_INT);
    }
    tf = (MPI_Wtime() - t0)/iters;
    MPI_Reduce(&tf, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) { printf("Standard comm: %8e\n", t0); }


    // Create Variables for send_procs/recv_procs 
    //    - Catch corner case with MPI_WEIGHTS_EMPTY
    int* send_procs  = A.send_comm.procs.data();
    if (A.send_comm.n_msgs == 0)
        send_procs = MPI_WEIGHTS_EMPTY;
    int* recv_procs = A.recv_comm.procs.data();
    if (A.recv_comm.n_msgs == 0)
        recv_procs = MPI_WEIGHTS_EMPTY;


    // Neighbor Collective 
    // 1. Create Topology Communicator
    MPI_Comm std_comm;
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < iters; i++) {
       MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD,
               A.recv_comm.n_msgs,
               recv_procs,
               MPI_UNWEIGHTED,
               A.send_comm.n_msgs, 
               send_procs,
               MPI_UNWEIGHTED,
               MPI_INFO_NULL, 
               0, 
               &std_comm);

       MPI_Neighbor_alltoallv(packed_send_vals.data(),
               A.send_comm.counts.data(),
               A.send_comm.ptr.data(),
               MPI_INT,
               neigh_recv_buffer.data(),
               A.recv_comm.counts.data(),
               A.recv_comm.ptr.data(),
               MPI_INT,
               std_comm);

       MPI_Comm_free(&std_comm);
       // Now flip
       MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD,
               A.send_comm.n_msgs,
               send_procs,
               MPI_UNWEIGHTED,
               A.recv_comm.n_msgs,
               recv_procs,
               MPI_UNWEIGHTED,
               MPI_INFO_NULL,
               0,
               &std_comm);

       MPI_Neighbor_alltoallv(neigh_recv_buffer.data(),
               A.recv_comm.counts.data(),
               A.recv_comm.ptr.data(),
               MPI_INT,
               packed_send_vals.data(),
               A.send_comm.counts.data(),
               A.send_comm.ptr.data(),
               MPI_INT,
               std_comm);

       MPI_Comm_free(&std_comm);
    }
    tf = (MPI_Wtime() - t0) / iters;
    MPI_Reduce(&tf, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) { printf("Standard graph create: %8e\n", t0); }

    // Error Checking
    for (int i = 0; i < A.recv_comm.size_msgs; i++)
    {
        if (std_recv_buffer[i] != neigh_recv_buffer[i])
        {
            printf("Rank %d recvd incorrect value! (MPI Version) \n", rank);
            break;
        }
    }
    for (int i = 0; i < A.send_comm.size_msgs; i++)
    {
        if (pure_packed_send[i] != packed_send_vals[i])
        {
            printf("Rank %d recvd incorrect (send) value! (MPI Version) \n", rank);
            break;
	}
    }

    // MPI Advance : Neighbor Collective 
    // 1. Create Topology Communicator
    MPIX_Topo* topo;
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < iters; i++) {
        MPIX_Topo_dist_graph_create_adjacent(
               A.recv_comm.n_msgs,
               recv_procs, 
               MPI_UNWEIGHTED,
               A.send_comm.n_msgs, 
               send_procs,
               MPI_UNWEIGHTED,
               MPI_INFO_NULL,
               false,
               &topo);
	
	MPIX_Neighbor_topo_alltoallv(packed_send_vals.data(),
               A.send_comm.counts.data(),
               A.send_comm.ptr.data(),
               MPI_INT,
               mpix_recv_buffer.data(),
               A.recv_comm.counts.data(),
               A.recv_comm.ptr.data(),
               MPI_INT,
               topo,
               MPI_COMM_WORLD);

        MPIX_Topo_free(topo);

	// Now do the flip!
	MPIX_Topo_dist_graph_create_adjacent(
               A.send_comm.n_msgs,
               send_procs,
               MPI_UNWEIGHTED,
               A.recv_comm.n_msgs,
               recv_procs,
               MPI_UNWEIGHTED,
               MPI_INFO_NULL,
               false,
               &topo);

        MPIX_Neighbor_topo_alltoallv(mpix_recv_buffer.data(),
               A.recv_comm.counts.data(),
               A.recv_comm.ptr.data(),
               MPI_INT,
               pure_packed_send.data(),
               A.send_comm.counts.data(),
               A.send_comm.ptr.data(),
               MPI_INT,
               topo,
               MPI_COMM_WORLD);

        MPIX_Topo_free(topo);
    }
    tf = (MPI_Wtime() - t0) / iters;
    MPI_Reduce(&tf, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) { printf("MPI advance graph create: %8e\n", t0); }

    // Error Checking
    for (int i = 0; i < A.recv_comm.size_msgs; i++)
    {
        if (std_recv_buffer[i] != mpix_recv_buffer[i])
        {
            printf("Rank %d recvd incorrect value (MPI Advance Version)!\n", rank);
            break;
        }
    }

    for (int i = 0; i < A.send_comm.size_msgs; i++)
    {
        if (pure_packed_send[i] != packed_send_vals[i])
        {
            printf("Rank %d recvd incorrect (send) value! (MPI Version) \n", rank);
            break;
        }
    }

	int reduce_data[4];

	reduce_data[0] = A.send_comm.counts.size();
	reduce_data[1] = A.recv_comm.counts.size();
	reduce_data[2] = std::accumulate(A.send_comm.counts.begin(),A.send_comm.counts.end(),0);
	reduce_data[3] = std::accumulate(A.recv_comm.counts.begin(),A.recv_comm.counts.end(),0);


	int max_data[4];
	int min_data[4];
	int sum_data[4];

	MPI_Reduce(reduce_data, max_data, 4, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(reduce_data, min_data, 4, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(reduce_data, sum_data, 4, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);


	if(rank == 0)
	{
		printf("Type Min Max Avg\n");
		printf("Send_Counts %d %d %f\n", min_data[0], max_data[0], 1.0*sum_data[0]/num_procs);
		printf("Recv_Counts %d %d %f\n", min_data[1], max_data[1], 1.0*sum_data[1]/num_procs);
		printf("Send_Size   %d %d %f\n", min_data[2], max_data[2], 1.0*sum_data[2]/num_procs);
		printf("Recv_Size   %d %d %f\n", min_data[3], max_data[3], 1.0*sum_data[3]/num_procs);
	}

    MPI_Finalize();
    return 0;
}
