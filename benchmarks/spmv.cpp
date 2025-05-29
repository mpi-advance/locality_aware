#include "mpi_advance.h"
#include "tests/sparse_mat.hpp"
#include "tests/par_binary_IO.hpp"
#include <numeric>
#include <cstring>
#include <cmath>
#include <ctime>

// Allows for Neighbor Alltoallv having different parameters
// than the locality version

/*
using Nstd = int (*) (const void*, const int*, const int*, MPI_Datatype, 
        void*, const int*, const int*, MPI_Datatype, MPIX_Comm*,
        MPIX_Info*, MPIX_Request**);
using Nloc = int (*) (const void*, const int*, const int*, const long*, MPI_Datatype, 
        void*, const int*, const int*, const long*, MPI_Datatype, MPIX_Comm*,
        MPIX_Info*, MPIX_Request**);
void neighbor_init(ParMat<int>& A, Nstd func, const void* sendbuf, const int* sendcounts, 
        const int* sdispls, MPI_Datatype sendtype, void* recvbuf, const int* recvcounts, 
        const int* rdispls, MPI_Datatype recvtype, MPIX_Comm* xcomm, 
        MPIX_Info* xinfo, MPIX_Request** xrequest_ptr)
{
    func(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype,
            xcomm, xinfo, xrequest_ptr);
}
void neighbor_init(ParMat<int>& A, Nloc func, const void* sendbuf, const int* sendcounts, 
        const int* sdispls, MPI_Datatype sendtype, void* recvbuf, const int* recvcounts, 
        const int* rdispls, MPI_Datatype recvtype, MPIX_Comm* xcomm, 
        MPIX_Info* xinfo, MPIX_Request** xrequest_ptr)
{
    std::vector<long> global_rows(A.local_rows);
    std::iota(global_rows.begin(), global_rows.end(), A.first_row);

    func(sendbuf, sendcounts, sdispls, global_rows.data(), sendtype, recvbuf, recvcounts, rdispls, 
            A.off_proc_columns.data(), recvtype, xcomm, xinfo, xrequest_ptr);
}*/

// Local SpMV
void spmv(double alpha, Mat& A, std::vector<double>& x, double beta, std::vector<double>& b)
{
    int start, end;
    double val;
    for (int i = 0; i < A.n_rows; i++)
    {
        start = A.rowptr[i];
        end = A.rowptr[i+1];
        val = 0;
        for (int j = start; j < end; j++)
        {
            val += alpha * A.data[j] * x[A.col_idx[j]];
        }
        b[i] = beta*b[i] + val;
    }
}

// Perform a single setup + n_spmvs iterations of SpMVs
// Times multiple iterations
template <typename F, typename N>
double time_spmvs(F discovery_func, N neighbor_func, ParMat<int>& A, std::vector<double>&x, std::vector<double>& x_dist, 
        std::vector<double>& b, MPIX_Comm* xcomm, int n_spmvs, int iterations)
{
    int n_sends, s_sends, proc;
    int *dest, *sendcounts, *sdispls;
    long *send_idx;

    std::vector<int> neigh_sdispls;
    std::vector<double> sendbuf;
    
    MPI_Barrier(xcomm->global_comm);
    double t0 = MPI_Wtime();
    for (int iter = 0; iter < iterations; iter++)
    {
        MPIX_Info* xinfo;
        MPIX_Info_init(&xinfo);

        MPIX_Comm* neighbor_comm;

        // Topology Discover
        s_sends = -1;
        discovery_func(A.recv_comm.n_msgs, A.recv_comm.size_msgs, A.recv_comm.procs.data(),
                    A.recv_comm.counts.data(), A.recv_comm.ptr.data(), MPI_LONG,
                A.off_proc_columns.data(),
                &n_sends, &s_sends, &dest, &sendcounts, &sdispls, MPI_LONG, (void**)&send_idx, xinfo, xcomm);
        for (int i = 0; i < s_sends; i++)
            send_idx[i] -= A.first_col;

        sendbuf.resize(s_sends);
        neigh_sdispls.resize(n_sends+1);
        neigh_sdispls[0] = 0;
        for (int i = 0; i < n_sends; i++)
            neigh_sdispls[i+1] = neigh_sdispls[i] + sendcounts[i];

        // Create neighbor communicator
        // TODO: Can replace with topology object version
        MPIX_Dist_graph_create_adjacent(MPI_COMM_WORLD,
                A.recv_comm.n_msgs,
                A.recv_comm.procs.data(),
                MPI_UNWEIGHTED,
                n_sends,
                dest, 
                MPI_UNWEIGHTED,
                MPI_INFO_NULL,
                0,
                &neighbor_comm);

        // Initialize neighbor alltoallv
        // TODO: Can replace with locality or part locality versions
        MPIX_Request* neighbor_request;
        neighbor_func(sendbuf.data(), sendcounts, sdispls, MPI_DOUBLE, 
            x_dist.data(), A.recv_comm.counts.data(), A.recv_comm.ptr.data(), MPI_DOUBLE, neighbor_comm, 
            xinfo, &neighbor_request);

        // Perform n_spmvs iterations of SpMVs
        MPI_Status status;
        for (int i = 0; i < n_spmvs; i++)
        {
            // Pack Data
            for (int i = 0; i < s_sends; i++)
                sendbuf[i] = x[send_idx[i]];

            // Start Communication
            MPIX_Start(neighbor_request);

            // Fully Local SpMV
            spmv(1.0, A.on_proc, x, 0.0, b);

            // Wait for Communication
            MPIX_Wait(neighbor_request, &status);

            // SpMV with Recvd Data
            spmv(1.0, A.off_proc, x_dist, 1.0, b);
        }

        // Free Neighbor Alltoallv Request
        MPIX_Request_free(&neighbor_request);

        // Free Neighbor Communciator
        MPIX_Comm_free(&neighbor_comm);
        
        MPIX_Info_free(&xinfo);

        MPIX_Free(dest);
        MPIX_Free(sendcounts);
        MPIX_Free(sdispls);
        MPIX_Free(send_idx);
    }
    double tfinal = (MPI_Wtime() - t0) / iterations;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, xcomm->global_comm);
    return t0;
} 


// Times the cost of SpMVs
// Calculates n_iterations for timer to take ~1sec
template <typename F, typename N>
double test_spmvs(F discovery_func, N neighbor_func, ParMat<int>& A, std::vector<double>& x, std::vector<double>& x_dist, 
        std::vector<double>& b, MPIX_Comm* xcomm, int n_spmvs)
{
    // Warm-Up
    time_spmvs(discovery_func, neighbor_func, A, x, x_dist, b, xcomm, n_spmvs, 1);

    // Time 2 Iterations
    double time = time_spmvs(discovery_func, neighbor_func, A, x, x_dist, b, xcomm, n_spmvs, 2);
    int n_iters = (1.0 / time) + 1;

    // Time N_iters Iterations
    time = time_spmvs(discovery_func, neighbor_func, A, x, x_dist, b, xcomm, n_spmvs, n_iters);
    return time;
}


// Benchmarks the different topology discovery methods 
// Along with variety of neighbor collectives
void benchmark_spmvs(ParMat<int>& A, std::vector<double>& x, std::vector<double>& x_dist, 
        std::vector<double>& b, MPIX_Comm* xcomm, int n_spmvs)
{
    int rank;
    MPI_Comm_rank(xcomm->global_comm, &rank);

    double time;

    std::vector<double> b_std(b.size());

    // Test Personalized + Neighbor
    time = test_spmvs(alltoallv_crs_personalized, MPIX_Neighbor_alltoallv_init,
            A, x, x_dist, b, xcomm, n_spmvs);
    if (rank == 0) printf("Personalized Standard Neighbor: %e\n", time);
    std::memcpy(b_std.data(), b.data(), b.size()*sizeof(double));
    
    // Test Nonblocking + Neighbor
    time = test_spmvs(alltoallv_crs_nonblocking, MPIX_Neighbor_alltoallv_init,
            A, x, x_dist, b, xcomm, n_spmvs);
    if (rank == 0) printf("Nonblocking Standard Neighbor: %e\n", time);
    for (int i = 0; i < b.size(); i++)
        if (fabs(b_std[i] - b[i]) > 1e-06)
        {
            printf("LOC: DIFFERENCE IN RESULTS! rank %d, i %d, %e vs %e\n", rank, i, b_std[i], b[i]);
            MPI_Abort(xcomm->global_comm, 1);
        }

    // Test Personalized Loc + Neighbor
    time = test_spmvs(alltoallv_crs_personalized_loc, MPIX_Neighbor_alltoallv_init,
            A, x, x_dist, b, xcomm, n_spmvs);
    if (rank == 0) printf("Personalized+Locality Standard Neighbor: %e\n", time);
    std::memcpy(b_std.data(), b.data(), b.size()*sizeof(double));

    // Test Nonblocking Loc + Neighbor
    time = test_spmvs(alltoallv_crs_nonblocking_loc, MPIX_Neighbor_alltoallv_init,
            A, x, x_dist, b, xcomm, n_spmvs);
    if (rank == 0) printf("Nonblocking+Locality Standard Neighbor: %e\n", time);
    std::memcpy(b_std.data(), b.data(), b.size()*sizeof(double));


}
    

// Must pass a suitesparse matrix as command line arg
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

    form_recv_comm(A);

    std::vector<double> x(A.on_proc.n_cols);
    std::vector<double> x_dist(A.off_proc.n_cols);
    std::vector<double> b(A.on_proc.n_rows);

    // Fill x with random values
    std::srand(unsigned(std::time(nullptr)));
    std::generate(x.begin(), x.end(), std::rand);

    MPIX_Comm* xcomm;
    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);

    int n_spmvs = 1000;
    benchmark_spmvs(A, x, x_dist, b, xcomm, n_spmvs);
    
    MPIX_Comm_free(&xcomm);

    MPI_Finalize();
    return 0;
}
