#include <numeric>
#include <ctime>
#include <random>

#include "locality_aware.h"
#include "par_binary_IO.hpp"
#include "sparse_mat.hpp"

void spmv(Mat& A, 
            std::vector<double>& x,
            std::vector<double>& b,
            double alpha,
            double beta)
{
    for (int i = 0; i < A.n_rows; i++)
    {
        double sum = 0;
        for (int j = A.rowptr[i]; j < A.rowptr[i+1]; j++)
            sum += A.data[j] * x[A.col_idx[j]];
        b[i] = alpha*sum + beta*b[i];
    }
}

void comm_init(ParMat<int>& A,
            std::vector<double>& sendbuf,
            std::vector<double>& recvbuf,
            std::vector<int>& send_order,
            std::vector<int>& recv_order,
            std::vector<MPI_Request>& send_req,
            std::vector<MPI_Request>& recv_req,
            MPIL_Comm* xcomm)
{
    int idx;
    int tag;
    MPIL_Comm_tag(xcomm, &tag);

    // Start communication
    for (int i = 0; i < A.recv_comm.n_msgs; i++)
    {
        idx = recv_order[i];
        MPI_Irecv(&(recvbuf[A.recv_comm.ptr[idx]]),
                    A.recv_comm.counts[idx],
                    MPI_DOUBLE,
                    A.recv_comm.procs[idx],
                    tag,
                    MPI_COMM_WORLD,
                    &(recv_req[i]));
    }
    for (int i = 0; i < A.send_comm.n_msgs; i++)
    {
        idx = send_order[i];
        MPI_Isend(&(sendbuf[A.send_comm.ptr[idx]]),
                    A.send_comm.counts[idx],
                    MPI_DOUBLE,
                    A.send_comm.procs[idx],
                    tag,
                    MPI_COMM_WORLD,
                    &(send_req[i]));
    }
}

void par_spmv(ParMat<int>& A,
            std::vector<double>& x,
            std::vector<double>& b,
            std::vector<double>& sendbuf,
            std::vector<double>& recvbuf,
            std::vector<int>& send_order,
            std::vector<int>& recv_order,
            std::vector<MPI_Request>& send_req,
            std::vector<MPI_Request>& recv_req,
            MPIL_Comm* xcomm)
{
    int idx;

    // Pack sendbuf
    for (int i = 0; i < A.send_comm.size_msgs; i++)
        sendbuf[i] = x[A.send_comm.idx[i]];

    comm_init(A, sendbuf, recvbuf, send_order, recv_order,
        send_req, recv_req, xcomm);

    // Fully local SpMV
    spmv(A.on_proc, x, b, 1.0, 0.0);

    // Wait for communication to finish
    MPI_Waitall(recv_req.size(), recv_req.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(send_req.size(), send_req.data(), MPI_STATUSES_IGNORE);

    // Off-process SpMV
    spmv(A.off_proc, recvbuf, b, 1.0, 1.0);
}

double time_par_spmv(ParMat<int>& A,
            std::vector<double>& x,
            std::vector<double>& b,
            std::vector<double>& sendbuf,
            std::vector<double>& recvbuf,
            std::vector<int>& send_order,
            std::vector<int>& recv_order,
            std::vector<MPI_Request>& send_req,
            std::vector<MPI_Request>& recv_req,
            MPIL_Comm* xcomm)
{
    double t0, tfinal;
    int n_iter = 1;

    // Warm-Up
    par_spmv(A, x, b, sendbuf, recvbuf, send_order, recv_order, 
            send_req, recv_req, xcomm);

    // Time single iteration
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    par_spmv(A, x, b, sendbuf, recvbuf, send_order, recv_order, 
            send_req, recv_req, xcomm);
    tfinal = MPI_Wtime() - t0;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (t0 < 0.1)
    {
        n_iter = 10;
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_iter; i++)
            par_spmv(A, x, b, sendbuf, recvbuf, send_order, recv_order, 
                    send_req, recv_req, xcomm);
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        n_iter = 1.0 / t0;
    }

    // Actual Timer
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
        par_spmv(A, x, b, sendbuf, recvbuf, send_order, recv_order, 
                send_req, recv_req, xcomm);
    tfinal = (MPI_Wtime() - t0) / n_iter;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    return t0;
}


void compare(std::vector<double>& recvbuf_std,
                std::vector<double>& recvbuf_new)
{
    for (int i = 0; i < recvbuf_std.size(); i++)
    {
        if (std::isnan(recvbuf_std[i]) && std::isnan(recvbuf_new[i]))
            continue;
        if (recvbuf_std[i] != recvbuf_new[i])
        {
            fprintf(stderr, "Difference at position %d, xorig %e, xnew %e\n", 
                    i, recvbuf_std[i], recvbuf_new[i]);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void reorder_recvs(ParMat<int>& A,
            std::vector<double>& sendbuf,
            std::vector<double>& recvbuf,
            std::vector<int>& send_order,
            std::vector<int>& recv_order,
            std::vector<MPI_Request>& send_req,
            std::vector<MPI_Request>& recv_req,
            MPIL_Comm* xcomm)
{
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    std::vector<int> procs_to_idx(num_procs);
    for (int i = 0; i < A.recv_comm.n_msgs; i++)
        procs_to_idx[A.recv_comm.procs[i]] = i;

    int idx;
    int tag;
    MPIL_Comm_tag(xcomm, &tag);
    
    // Start sends
    for (int i = 0; i < A.send_comm.n_msgs; i++)
    {
        idx = send_order[i];
        MPI_Isend(&(sendbuf[A.send_comm.ptr[idx]]),
                    A.send_comm.counts[idx],
                    MPI_DOUBLE,
                    A.send_comm.procs[idx],
                    tag,
                    MPI_COMM_WORLD,
                    &(send_req[i]));
    }

    MPI_Status status;
    for (int i = 0; i < recv_req.size(); i++)
    {
        MPI_Probe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);
        int proc = status.MPI_SOURCE;
        idx = procs_to_idx[proc];
        MPI_Recv(&(recvbuf[A.recv_comm.ptr[idx]]),
                    A.recv_comm.counts[idx],
                    MPI_DOUBLE, 
                    A.recv_comm.procs[idx],
                    tag,
                    MPI_COMM_WORLD,
                    &status);
        recv_order[i] = idx;
    }

    MPI_Waitall(send_req.size(), send_req.data(), MPI_STATUSES_IGNORE);
}

void reorder_sends(ParMat<int>& A,
            std::vector<double>& sendbuf,
            std::vector<double>& recvbuf,
            std::vector<int>& send_order,
            std::vector<int>& recv_order,
            std::vector<MPI_Request>& send_req,
            std::vector<MPI_Request>& recv_req,
            MPIL_Comm* xcomm)
{
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    std::vector<int> procs_to_idx(num_procs);
    for (int i = 0; i < A.send_comm.n_msgs; i++)
        procs_to_idx[A.send_comm.procs[i]] = i;

    int idx;
    int tag;
    MPIL_Comm_tag(xcomm, &tag);


    for (int i = 0; i < A.recv_comm.n_msgs; i++)
    {   
        idx = recv_order[i];
        MPI_Isend(&(recvbuf[A.recv_comm.ptr[idx]]),
                    A.recv_comm.counts[idx],
                    MPI_DOUBLE,
                    A.recv_comm.procs[idx],
                    tag,
                    MPI_COMM_WORLD,
                    &(recv_req[i]));
    }


    MPI_Status status;
    for (int i = 0; i < send_req.size(); i++)
    {
        MPI_Probe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);
        int proc = status.MPI_SOURCE;
        idx = procs_to_idx[proc];
        MPI_Recv(&(sendbuf[A.send_comm.ptr[idx]]),
                    A.send_comm.counts[idx],
                    MPI_DOUBLE,
                    A.send_comm.procs[idx],
                    tag,
                    MPI_COMM_WORLD,
                    &status);
        send_order[i] = idx;
    }

    MPI_Waitall(recv_req.size(), recv_req.data(), MPI_STATUSES_IGNORE);
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    MPIL_Init(MPI_COMM_WORLD);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc == 1)
    {
        if (rank == 0)
        {
            printf("Pass Matrix Filename as Command Line Arg!\n");
        }
        MPI_Finalize();
        return 1;
    }
    char* filename = argv[1];

    // Read suitesparse matrix
    ParMat<int> A;
    int file_error = readParMatrix(filename, A);
    if (file_error)
    {
        return 1;
    }

    // Form Communication Package (A.send_comm, A.recv_comm)
    form_comm(A);

    double t0;

    std::vector<double> x(A.on_proc.n_rows);
    std::vector<double> b(A.on_proc.n_rows);
    std::vector<double> b_new(A.on_proc.n_rows);
    std::generate(x.begin(), x.end(), 
        []() { return (double)rand() / RAND_MAX; });

    std::vector<double> sendbuf(A.send_comm.size_msgs);
    std::vector<double> recvbuf(A.recv_comm.size_msgs);

    MPIL_Comm* xcomm;
    MPIL_Comm_init(&xcomm, MPI_COMM_WORLD);

    std::vector<MPI_Request> send_req(A.send_comm.n_msgs);
    std::vector<MPI_Request> recv_req(A.recv_comm.n_msgs);

    std::vector<int> send_order(A.send_comm.n_msgs);
    std::vector<int> recv_order(A.recv_comm.n_msgs);

    // Standard Communication
    std::iota(send_order.begin(), send_order.end(), 0);
    std::iota(recv_order.begin(), recv_order.end(), 0);
    for (int iter = 0; iter < 5; iter++)
    {
        t0 = time_par_spmv(A, x, b, sendbuf, recvbuf, 
                send_order, recv_order, send_req, recv_req, xcomm);
        if (rank == 0) printf("Iter %d: Original SpMV Time: %e\n", iter, t0);
    }

    // Reorder Recvs
    std::iota(send_order.begin(), send_order.end(), 0);
    std::iota(recv_order.begin(), recv_order.end(), 0);
    reorder_recvs(A, sendbuf, recvbuf, send_order, recv_order,
            send_req, recv_req, xcomm);
    for (int iter = 0; iter < 5; iter++)
    {
        t0 = time_par_spmv(A, x, b_new, sendbuf, recvbuf, 
                send_order, recv_order, send_req, recv_req, xcomm);
        if (rank == 0) printf("Iter %d: Reordered Recvs SpMV Time: %e\n", iter, t0);
    }
    compare(b, b_new);

    // Reorder Sends and Recvs
    std::fill(b_new.begin(), b_new.end(), 0);
    std::iota(send_order.begin(), send_order.end(), 0);
    std::iota(recv_order.begin(), recv_order.end(), 0);
    reorder_sends(A, sendbuf, recvbuf, send_order, recv_order,
            send_req, recv_req, xcomm);
    reorder_recvs(A, sendbuf, recvbuf, send_order, recv_order,
            send_req, recv_req, xcomm);
    for (int iter = 0; iter < 5; iter++)
    {
        t0 = time_par_spmv(A, x, b_new, sendbuf, recvbuf, 
                send_order, recv_order, send_req, recv_req, xcomm);
        if (rank == 0) printf("Iter %d: Reordered Sends/Recvs SpMV Time: %e\n", iter, t0);
    }
    compare(b, b_new);

    MPIL_Comm_free(&xcomm);

    MPIL_Finalize();
    MPI_Finalize();
    return 0;
}
