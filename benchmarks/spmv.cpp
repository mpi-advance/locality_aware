#include "mpi_advance.h"
#include "tests/sparse_mat.hpp"
#include "tests/par_binary_IO.hpp"

template <typename T>
void spmv(double alpha, Mat& A, std::vector<T>& x, double beta, std::vector<T>& b)
{
    int start, end;
    T val;
    for (int i = 0; i < A.n_rows; i++)
    {
  		start = A.rowptr[i];
		end = A.rowptr[i+1];
		val = 0;
		for (int j = start; j < end; j++)
		{
			col_idx = A.col_idx[j];
			val += alpha * A.data[j] * x[col_idx];
        }
		b[i] = beta*b[i] + val;
}


template <typename F, typename N, typename T>
double time_spmvs(F topology_discovery_alg, N neighbor_collective_alg, ParMat<int>& A,
        std::vector<T>&x, std::vector<T>& x_dist, std::vector<T>& b, int n_timings, int n_spmvs_per_timing)
{
    int n_recvs, s_recvs, proc;
    int *src, *recvcounts, *rdispls;
    long *recvvals;
    for (int iter = 0; iter < n_timings; iter++)
	{
        MPIX_Info* xinfo;
        MPIX_Info_init(&xinfo);

		MPIX_Comm neighbor_comm;

		// Topology Discover
		s_recvs = -1;
		F(A.recv_comm.n_msgs, A.recv_comm.size_msgs, A.recv_comm.procs.data(),
       		     A.recv_comm.counts.data(), A.recv_comm.ptr.data(), MPI_LONG,
            	A.off_proc_columns.data(),
            	&n_recvs, &s_recvs, &src, &recvcounts, &rdispls, MPI_LONG, (void**)&recvvals, xinfo, xcomm);
		for (int i = 0; i < s_recvs; i++)
			recvvals[i] -= A.first_col;
        // n_recvs = A.send_comm.n_msgs
        // s_recvs = A.send_comm.size_msgs
        // src = A.send_comm.procs
        // recvcounts = A.send_comm.counts
        // rdispls = A.send_comm.displs
        // recvvals = A.send_comm.idx


		// Create neighbor communicator
        MPIX_Dist_graph_create_adjacent(MPI_COMM_WORLD,
                A.recv_comm.n_msgs,
                A.recv_comm.procs.data(),
                MPI_UNWEIGHTED,
				n_recvs,
				src, 
                MPI_UNWEIGHTED,
                MPI_INFO_NULL,
                0,
                &neighbor_comm);

		// Initialize neighbor alltoallv
    	MPIX_Neighbor_alltoallv_init(x.data(),
				recvcounts,
				rdispls,
       	    	MPI_INT,
            	x_dist.data(),
            	A.recv_comm.counts.data(),
            	A.recv_comm.ptr.data(),
            	MPI_INT,
            	neighbor_comm,
            	xinfo,
            	&neighbor_request);

		// Perform n_spmvs iterations of SpMVs
		for (int i = 0; i < n_spmvs; i++)
		{
			// Start Communication
    		MPIX_Start(neighbor_request);

			// Fully Local SpMV
			spmv(1.0, A.on_proc, x.data(), 0.0, b.data());

			// Wait for Communication
    		MPIX_Wait(neighbor_request, &status);

			// SpMV with Recvd Data
			spmv(1.0, A.off_proc, x_dist.data(), 1.0, b.data());
		}

    	MPIX_Request_free(&neighbor_request);





    	MPIX_Free(src);
    	MPIX_Free(recvcounts);
    	MPIX_Free(rdispls);
    	MPIX_Free(recvvals);
	}
} 

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double t0, tfinal;
    
    int n_iter = 10;
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
    update_locality(xcomm, 4);

    int n_recvs, s_recvs, proc;
    int *src, *rdispls, *recvcounts;
    long* recvvals;

    std::vector<int> proc_count(num_procs, -1);
    std::vector<int> proc_displs(num_procs, -1);
    std::vector<long> orig_indices(A.send_comm.size_msgs+1);
    for (int i = 0; i < A.send_comm.n_msgs; i++)
    {
        proc = A.send_comm.procs[i];
        proc_count[proc] = A.send_comm.counts[i];
        proc_displs[proc] = A.send_comm.ptr[i];
    }
    for (int i = 0; i < A.send_comm.size_msgs; i++)
        orig_indices[i] = A.send_comm.idx[i] + A.first_col;

	// Time Personalized
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        s_recvs = -1;
        alltoallv_crs_personalized(A.recv_comm.n_msgs, A.recv_comm.size_msgs, A.recv_comm.procs.data(),
                A.recv_comm.counts.data(), A.recv_comm.ptr.data(), MPI_LONG,
                A.off_proc_columns.data(), &n_recvs, &s_recvs, &src, &recvcounts,
				&rdispls, MPI_LONG, (void**) &recvvals, xinfo, xcomm);
    }
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("MPI_Alltoall_crs Time (Personalized VERSION): %e\n", t0/n_iter);
    compare(n_recvs, s_recvs, src, recvcounts, rdispls, recvvals,
            A.send_comm.n_msgs, A.send_comm.size_msgs, proc_count.data(), proc_displs.data(),
            orig_indices.data());
    MPIX_Free(src);
    MPIX_Free(recvcounts);
    MPIX_Free(rdispls);
    MPIX_Free(recvvals);

    // Time Nonblocking
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        s_recvs = -1;
        alltoallv_crs_nonblocking(A.recv_comm.n_msgs, A.recv_comm.size_msgs, A.recv_comm.procs.data(),
                A.recv_comm.counts.data(), A.recv_comm.ptr.data(), MPI_LONG,
                A.off_proc_columns.data(), &n_recvs, &s_recvs, &src, &recvcounts,
				&rdispls, MPI_LONG, (void**)&recvvals, xinfo, xcomm);
    }
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("MPI_Alltoall_crs Time (Personalized VERSION): %e\n", t0/n_iter);
    compare(n_recvs, s_recvs, src, recvcounts, rdispls, recvvals,
            A.send_comm.n_msgs, A.send_comm.size_msgs, proc_count.data(), proc_displs.data(),
            orig_indices.data());
    MPIX_Free(src);
    MPIX_Free(recvcounts);
    MPIX_Free(rdispls);
    MPIX_Free(recvvals);

    // Time Personalized Locality
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        s_recvs = -1;
        alltoallv_crs_personalized_loc(A.recv_comm.n_msgs, A.recv_comm.size_msgs, A.recv_comm.procs.data(),
                A.recv_comm.counts.data(), A.recv_comm.ptr.data(), MPI_LONG,
                A.off_proc_columns.data(), &n_recvs, &s_recvs, &src, &recvcounts,
				&rdispls, MPI_LONG, (void**) &recvvals, xinfo, xcomm);
    }
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("MPI_Alltoall_crs Time (Personalized VERSION): %e\n", t0/n_iter);
    compare(n_recvs, s_recvs, src, recvcounts, rdispls, recvvals,
            A.send_comm.n_msgs, A.send_comm.size_msgs, proc_count.data(), proc_displs.data(),
            orig_indices.data());
    MPIX_Free(src);
    MPIX_Free(recvcounts);
    MPIX_Free(rdispls);
    MPIX_Free(recvvals);

    // Time Nonblocking Locality
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        s_recvs = -1;
        alltoallv_crs_nonblocking_loc(A.recv_comm.n_msgs, A.recv_comm.size_msgs, A.recv_comm.procs.data(),
                A.recv_comm.counts.data(), A.recv_comm.ptr.data(), MPI_LONG,
                A.off_proc_columns.data(), &n_recvs, &s_recvs, &src, &recvcounts,
				&rdispls, MPI_LONG, (void**) &recvvals, xinfo, xcomm);
    }
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("MPI_Alltoall_crs Time (Personalized VERSION): %e\n", t0/n_iter);
    compare(n_recvs, s_recvs, src, recvcounts, rdispls, recvvals,
            A.send_comm.n_msgs, A.send_comm.size_msgs, proc_count.data(), proc_displs.data(),
            orig_indices.data());
    MPIX_Free(src);
    MPIX_Free(recvcounts);
    MPIX_Free(rdispls);
    MPIX_Free(recvvals);

    
    MPIX_Info_free(&xinfo);
    MPIX_Comm_free(&xcomm);

    MPI_Finalize();
    return 0;
}
