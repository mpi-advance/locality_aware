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

    double t0, tfinal;
    
    int n_iter = 1000;
    if(num_procs > 9000)
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

    // Form MPIX_Comm initial communicator (should be cheap)
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);
        MPIX_Comm_free(xcomm);
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
    std::vector<int> src(A.send_comm.n_msgs+1);
    std::vector<int> recvvals(A.send_comm.n_msgs+1);
    std::vector<int> recvcounts(num_procs, 0);


	// Time RMA
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        n_recvs = -1;
        alltoall_crs_rma(A.recv_comm.n_msgs, A.recv_comm.procs.data(), 1, MPI_INT,
                A.recv_comm.counts.data(), &n_recvs, src.data(), 1, MPI_INT,
                recvvals.data(), xcomm);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("MPI_Alltoall_crs Time (RMA VERSION): %e\n", t0/n_iter);

	// Time Personalized
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        n_recvs = -1;
        alltoall_crs_personalized(A.recv_comm.n_msgs, A.recv_comm.procs.data(), 1, MPI_INT,
                A.recv_comm.counts.data(), &n_recvs, src.data(), 1, MPI_INT,
                recvvals.data(), xcomm);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("MPI_Alltoall_crs Time (Personalized VERSION): %e\n", t0/n_iter);

	// Time Nonblocking
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        n_recvs = -1;
        alltoall_crs_nonblocking(A.recv_comm.n_msgs, A.recv_comm.procs.data(), 1, MPI_INT,
                A.recv_comm.counts.data(), &n_recvs, src.data(), 1, MPI_INT,
                recvvals.data(), xcomm);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("MPI_Alltoall_crs Time (Nonblocking VERSION): %e\n", t0/n_iter);


	// Time Personalized Locality
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        n_recvs = -1;
        alltoall_crs_personalized_loc(A.recv_comm.n_msgs, A.recv_comm.procs.data(), 1, MPI_INT,
                A.recv_comm.counts.data(), &n_recvs, src.data(), 1, MPI_INT,
                recvvals.data(), xcomm);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("MPI_Alltoall_crs Time (Personalized Locality VERSION): %e\n", t0/n_iter);

	// Time Nonblocking Locality
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        n_recvs = -1;
        alltoall_crs_nonblocking_loc(A.recv_comm.n_msgs, A.recv_comm.procs.data(), 1, MPI_INT,
                A.recv_comm.counts.data(), &n_recvs, src.data(), 1, MPI_INT,
                recvvals.data(), xcomm);
    }
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("MPI_Alltoall_crs Time (Nonblocking Locality VERSION): %e\n", t0/n_iter);
    
    MPIX_Comm_free(xcomm);

    MPI_Finalize();
    return 0;
}
