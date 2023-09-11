#include "mpi_advance.h"
#include "tests/sparse_mat.hpp"
#include "tests/par_binary_IO.hpp"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

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
    form_comm(A);

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
    if (A.send_comm.size_msgs)
    {
        packed_send_vals.resize(A.send_comm.size_msgs);
        for (int i = 0; i < A.send_comm.size_msgs; i++)
        {
            packed_send_vals[i] = send_vals[A.send_comm.idx[i]];
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
    communicate(A, send_vals, std_recv_buffer, MPI_INT);


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

    // 2. Call Neighbor Alltoallv
    MPI_Neighbor_alltoallv(packed_send_vals.data(), 
            A.send_comm.counts.data(),
            A.send_comm.ptr.data(), 
            MPI_INT,
            neigh_recv_buffer.data(), 
            A.recv_comm.counts.data(),
            A.recv_comm.ptr.data(), 
            MPI_INT,
            std_comm);
    // 3. Free Topology Communicator
    MPI_Comm_free(&std_comm);

    // Error Checking
    for (int i = 0; i < A.recv_comm.size_msgs; i++)
    {
        if (std_recv_buffer[i] != neigh_recv_buffer[i])
        {
            printf("Rank %d recvd incorrect value! (MPI Version) \n", rank);
            break;
        }
    }



    // MPI Advance : Neighbor Collective 
    // 1. Create Topology Communicator
    MPIX_Comm* xcomm;
    MPIX_Dist_graph_create_adjacent(MPI_COMM_WORLD,
            A.recv_comm.n_msgs,
            recv_procs, 
            MPI_UNWEIGHTED,
            A.send_comm.n_msgs, 
            send_procs,
            MPI_UNWEIGHTED,
            MPI_INFO_NULL, 
            0, 
            &xcomm);
    // 2. Call Neighbor Alltoallv
    MPIX_Neighbor_alltoallv(packed_send_vals.data(), 
            A.send_comm.counts.data(),
            A.send_comm.ptr.data(), 
            MPI_INT,
            mpix_recv_buffer.data(), 
            A.recv_comm.counts.data(),
            A.recv_comm.ptr.data(), 
            MPI_INT,
            xcomm);
    // 3. Free Topology Communicator
    MPIX_Comm_free(xcomm);

    // Error Checking
    for (int i = 0; i < A.recv_comm.size_msgs; i++)
    {
        if (std_recv_buffer[i] != mpix_recv_buffer[i])
        {
            printf("Rank %d recvd incorrect value (MPI Advance Version)!\n", rank);
            break;
        }
    }

    MPI_Finalize();
    return 0;
}
