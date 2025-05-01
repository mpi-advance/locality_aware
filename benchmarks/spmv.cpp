#include "mpi_advance.h"
#include "tests/sparse_mat.hpp"
#include "tests/par_binary_IO.hpp"
#include <numeric>

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
}

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

template <typename F, typename N>
double time_spmvs(F discovery_func, N neighbor_func, ParMat<int>& A, std::vector<double>&x, std::vector<double>& x_dist, 
        std::vector<double>& b, MPIX_Comm* xcomm, int n_spmvs, int iterations)
{
    int n_recvs, s_recvs, proc;
    int *src, *recvcounts, *rdispls;
    long *recvvals;
    
    MPI_Barrier(xcomm->global_comm);
    double t0 = MPI_Wtime();
    for (int iter = 0; iter < iterations; iter++)
    {
        MPIX_Info* xinfo;
        MPIX_Info_init(&xinfo);

        MPIX_Comm* neighbor_comm;

        // Topology Discover
        s_recvs = -1;
        discovery_func(A.recv_comm.n_msgs, A.recv_comm.size_msgs, A.recv_comm.procs.data(),
                    A.recv_comm.counts.data(), A.recv_comm.ptr.data(), MPI_LONG,
                A.off_proc_columns.data(),
                &n_recvs, &s_recvs, &src, &recvcounts, &rdispls, MPI_LONG, (void**)&recvvals, xinfo, xcomm);
        for (int i = 0; i < s_recvs; i++)
            recvvals[i] -= A.first_col;

        // Create neighbor communicator
        // TODO: Can replace with topology object version
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
        // TODO: Can replace with locality or part locality versions
        MPIX_Request* neighbor_request;
        neighbor_init(A, neighbor_func, x.data(), recvcounts, rdispls, MPI_INT, 
            x_dist.data(), A.recv_comm.counts.data(), A.recv_comm.ptr.data(), MPI_INT, neighbor_comm, 
            xinfo, &neighbor_request);

        // Perform n_spmvs iterations of SpMVs
        MPI_Status status;
        for (int i = 0; i < n_spmvs; i++)
        {
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

        MPIX_Free(src);
        MPIX_Free(recvcounts);
        MPIX_Free(rdispls);
        MPIX_Free(recvvals);
    }
    double tfinal = (MPI_Wtime() - t0) / iterations;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, xcomm->global_comm);
    return t0;
} 

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

void benchmark_spmvs(ParMat<int>& A, std::vector<double>& x, std::vector<double>& x_dist, 
        std::vector<double>& b, MPIX_Comm* xcomm, int n_spmvs)
{
    int rank;
    MPI_Comm_rank(xcomm->global_comm, &rank);

    double time;

    std::vector<double> b_std(b.size());

    // Loop through all topology discovery methods
    using F = int (*)(int, int, int*, int*, int*, MPI_Datatype, 
            void*, int*, int*, int**, int**, int**, MPI_Datatype,
            void**, MPIX_Info*, MPIX_Comm*);
    std::vector<F> topology_discovery_funcs = {alltoallv_crs_personalized,
            alltoallv_crs_nonblocking,
            alltoallv_crs_personalized_loc,
            alltoallv_crs_nonblocking_loc};
    std::vector<const char*> discovery_names = {"Personalized", "Nonblocking", "Personalized + Locality", 
            "Nonblocking + Locality"};

    for (int idx = 0; idx < topology_discovery_funcs.size(); idx++)
    {
        // Time Standard Neighbor Collective SpMV
        time = test_spmvs(topology_discovery_funcs[idx], MPIX_Neighbor_alltoallv_init, 
                A, x, x_dist, b, xcomm, n_spmvs);
        if (rank == 0) printf("%s, Standard Neighbor: %e\n", discovery_names[idx], time);
        std::memcpy(b_std.data(), b.data(), b.size()*sizeof(double));

        // Test Full Locality-Aware SpMV
        time = test_spmvs(topology_discovery_funcs[idx], MPIX_Neighbor_locality_alltoallv_init, 
                A, x, x_dist, b, xcomm, n_spmvs);
        if (rank == 0) printf("%s, Locality-Aware Neighbor: %e\n", discovery_names[idx], time);
        for (int i = 0; i < b.size(); i++)
            if (fabs(b_std[i] - b[i]) > 1e-06)
            {
                printf("DIFFERENCE IN RESULTS!\n");
                MPI_Abort(xcomm->global_comm, 1);
            }

        // Test Partial Locality-Aware SpMV (doesn't remove duplicates)
        time = test_spmvs(topology_discovery_funcs[idx], MPIX_Neighbor_part_locality_alltoallv_init, 
                A, x, x_dist, b, xcomm, n_spmvs);
        if (rank == 0) printf("%s, Part Locality Neighbor: %e\n", discovery_names[idx], time);
        for (int i = 0; i < b.size(); i++)
            if (fabs(b_std[i] - b[i]) > 1e-06)
            {
                printf("DIFFERENCE IN RESULTS!\n");
                MPI_Abort(xcomm->global_comm, 2);
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
