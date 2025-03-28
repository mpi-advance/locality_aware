#include "mpi_advance.h"
#include "tests/sparse_mat.hpp"
#include "tests/par_binary_IO.hpp"

#include <numeric>

void compare(int n_recvs, int* src, int* counts, int orig_n_recvs, int* orig_proc_counts)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (n_recvs != orig_n_recvs)
    {
        printf("Num Messages Incorrect! Rank %d got %d, should be %d\n", 
                rank, n_recvs, orig_n_recvs);
		return;
    }

    for (int i = 0; i < n_recvs; i++)
    {
        if (orig_proc_counts[src[i]] != counts[i])
        {
            printf("Rank %d, msgcounts from proc %d incorrect!  Got %d, should be %d\n",
                    rank, src[i], orig_proc_counts[src[i]], counts[i]);
            break;
        }
    }
}


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double t0, tfinal;
    
    int n_iter = 1000;
    if(num_procs > 1000)
	    n_iter = 100;

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

    MPIX_Comm* xcomm;

    MPIX_Info* xinfo;
    MPIX_Info_init(&xinfo);

    // Form MPIX_Comm initial communicator (should be cheap)
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);
        MPIX_Comm_free(&xcomm);
    }
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("MPIX_Comm_init time %e\n", t0/n_iter);

    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);

	// Split node communicator
	MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        MPIX_Comm_topo_init(xcomm);
        MPIX_Comm_topo_free(xcomm);
    }
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("MPIX_Comm_topo_init time %e\n", t0/n_iter);

    MPIX_Comm_topo_init(xcomm);

    // Form Window
    int bytes = num_procs * sizeof(int);
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        MPIX_Comm_win_init(xcomm, bytes, sizeof(int));
        MPIX_Comm_win_free(xcomm);
    }
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("MPIX_Comm_win_init_time %e\n", t0/n_iter);

    MPIX_Comm_win_init(xcomm, bytes, sizeof(int));

    int n_recvs;
    int *src, *recvvals;

    std::vector<int> proc_count(num_procs, -1);
    for (int i = 0; i < A.send_comm.n_msgs; i++)
        proc_count[A.send_comm.procs[i]] = A.send_comm.counts[i];


	// Time RMA
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        n_recvs = -1;
        alltoall_crs_rma(A.recv_comm.n_msgs, A.recv_comm.procs.data(), 1, MPI_INT,
                A.recv_comm.counts.data(), &n_recvs, &src, 1, MPI_INT,
                (void**)&recvvals, xinfo, xcomm);
    }
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("MPI_Alltoall_crs Time (RMA VERSION): %e\n", t0/n_iter);
    compare(n_recvs, src, recvvals, A.send_comm.n_msgs, proc_count.data());
    MPIX_Free(src);
    MPIX_Free(recvvals);

	// Time Personalized
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        n_recvs = -1;
        alltoall_crs_personalized(A.recv_comm.n_msgs, A.recv_comm.procs.data(), 1, MPI_INT,
                A.recv_comm.counts.data(), &n_recvs, &src, 1, MPI_INT,
                (void**)&recvvals, xinfo, xcomm);
    }
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("MPI_Alltoall_crs Time (RMA VERSION): %e\n", t0/n_iter);
    compare(n_recvs, src, recvvals, A.send_comm.n_msgs, proc_count.data());
    MPIX_Free(src);
    MPIX_Free(recvvals);

	// Time Nonblocking
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        n_recvs = -1;
        alltoall_crs_nonblocking(A.recv_comm.n_msgs, A.recv_comm.procs.data(), 1, MPI_INT,
                A.recv_comm.counts.data(), &n_recvs, &src, 1, MPI_INT,
                (void**)&recvvals, xinfo, xcomm);
    }
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("MPI_Alltoall_crs Time (RMA VERSION): %e\n", t0/n_iter);
    compare(n_recvs, src, recvvals, A.send_comm.n_msgs, proc_count.data());
    MPIX_Free(src);
    MPIX_Free(recvvals);


	// Time Personalized Locality
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        n_recvs = -1;
        alltoall_crs_personalized_loc(A.recv_comm.n_msgs, A.recv_comm.procs.data(), 1, MPI_INT,
                A.recv_comm.counts.data(), &n_recvs, &src, 1, MPI_INT,
                (void**)&recvvals, xinfo, xcomm);
    }
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("MPI_Alltoall_crs Time (RMA VERSION): %e\n", t0/n_iter);
    compare(n_recvs, src, recvvals, A.send_comm.n_msgs, proc_count.data());
    MPIX_Free(src);
    MPIX_Free(recvvals);

	// Time Nonblocking Locality
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        n_recvs = -1;
        alltoall_crs_nonblocking_loc(A.recv_comm.n_msgs, A.recv_comm.procs.data(), 1, MPI_INT,
                A.recv_comm.counts.data(), &n_recvs, &src, 1, MPI_INT,
                (void**)&recvvals, xinfo, xcomm);
    }
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("MPI_Alltoall_crs Time (RMA VERSION): %e\n", t0/n_iter);
    compare(n_recvs, src, recvvals, A.send_comm.n_msgs, proc_count.data());
    MPIX_Free(src);
    MPIX_Free(recvvals);

    MPIX_Info_free(&xinfo);
    MPIX_Comm_free(&xcomm);
    MPI_Finalize();
    return 0;
}
