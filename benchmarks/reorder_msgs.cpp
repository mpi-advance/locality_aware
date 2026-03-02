#include <numeric>
#include <ctime>
#include <random>

#include "locality_aware.h"
#include "par_binary_IO.hpp"
#include "sparse_mat.hpp"

void compare(std::vector<double>& recvbuf_std,
                std::vector<double>& recvbuf_new)
{
    for (int i = 0; i < recvbuf_std.size(); i++)
    {
        if (recvbuf_std[i] != recvbuf_new[i])
        {
            fprintf(stderr, "Difference at position %d, xorig %e, xnew %e\n", 
                    i, recvbuf_std[i], recvbuf_new[i]);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void comm_init(ParMat<int>& A,
                std::vector<double>& sendbuf,
                std::vector<double>& recvbuf,
                std::vector<MPI_Request>& send_req,
                std::vector<MPI_Request>& recv_req,
                MPIL_Comm* xcomm)
{
    int tag;
    MPIL_Comm_tag(xcomm, &tag);

    recv_req.resize(A.recv_comm.n_msgs);
    for (int i = 0; i < A.recv_comm.n_msgs; i++)
    {
        MPI_Recv_init(&(recvbuf[A.recv_comm.ptr[i]]),
                      A.recv_comm.counts[i],
                      MPI_DOUBLE, 
                      A.recv_comm.procs[i],
                      tag,
                      MPI_COMM_WORLD, 
                      &(recv_req[i]));
    }

    send_req.resize(A.send_comm.n_msgs);
    for (int i = 0; i < A.send_comm.n_msgs; i++)
    {
        MPI_Send_init(&(sendbuf[A.send_comm.ptr[i]]),
                      A.send_comm.counts[i],
                      MPI_DOUBLE,
                      A.send_comm.procs[i],
                      tag,
                      MPI_COMM_WORLD, 
                      &(send_req[i]));
    }
}

void reverse_comm_init(ParMat<int>& A,
                std::vector<double>& sendbuf,
                std::vector<double>& recvbuf,
                std::vector<MPI_Request>& send_req,
                std::vector<MPI_Request>& recv_req,
                MPIL_Comm* xcomm)
{
    int tag;
    MPIL_Comm_tag(xcomm, &tag);

    send_req.resize(A.recv_comm.n_msgs);
    for (int i = 0; i < A.recv_comm.n_msgs; i++)
    {
        MPI_Send_init(&(recvbuf[A.recv_comm.ptr[i]]),
                      A.recv_comm.counts[i],
                      MPI_DOUBLE,
                      A.recv_comm.procs[i],
                      tag,
                      MPI_COMM_WORLD, 
                      &(send_req[i]));
    }

    recv_req.resize(A.send_comm.n_msgs);
    for (int i = 0; i < A.send_comm.n_msgs; i++)
    {
        MPI_Recv_init(&(sendbuf[A.send_comm.ptr[i]]),
                      A.send_comm.counts[i],
                      MPI_DOUBLE,
                      A.send_comm.procs[i],
                      tag,
                      MPI_COMM_WORLD, 
                      &(recv_req[i]));
    }
}

std::vector<int> find_order(std::vector<MPI_Request>& send_req,
                    std::vector<MPI_Request>& recv_req)
{
    MPI_Startall(recv_req.size(), recv_req.data());
    MPI_Startall(send_req.size(), send_req.data());

    int idx;
    std::vector<int> order(recv_req.size());
    for (int i = 0; i < recv_req.size(); i++)
    {
        MPI_Waitany(recv_req.size(), recv_req.data(), &idx, MPI_STATUS_IGNORE);
        order[i] = idx;
    }

    MPI_Waitall(send_req.size(), send_req.data(), MPI_STATUSES_IGNORE);

    return order;
}

void reorder_recvs(std::vector<MPI_Request>& send_req, 
                    std::vector<MPI_Request>& recv_req)
{
    std::vector<int> recv_order = find_order(send_req, recv_req);
    std::vector<MPI_Request> recv_ordered(recv_req.size());
    for (int i = 0; i < recv_req.size(); i++)
        recv_ordered[i] = recv_req[recv_order[i]];
    recv_req = recv_ordered;
}

void reorder_comm(std::vector<MPI_Request>& send_req,
                    std::vector<MPI_Request>& recv_req,
                    std::vector<MPI_Request>& reverse_send_req,
                    std::vector<MPI_Request>& reverse_recv_req)
{
    std::vector<int> send_order = find_order(
                reverse_send_req, reverse_recv_req);
    std::vector<MPI_Request> send_ordered(send_req.size());
    for (int i = 0; i < send_req.size(); i++)
        send_ordered[i] = send_req[send_order[i]];
    send_req = send_ordered;

    std::vector<int> recv_order = find_order(send_req, recv_req);
    std::vector<MPI_Request> recv_ordered(recv_req.size());
    for (int i = 0; i < recv_req.size(); i++)
        recv_ordered[i] = recv_req[recv_order[i]];
    recv_req = recv_ordered;
}

void free_requests(std::vector<MPI_Request>& send_req,
                    std::vector<MPI_Request>& recv_req)
{
    for (int i = 0; i < recv_req.size(); i++)
        MPI_Request_free(&recv_req[i]);

    for (int i = 0; i < send_req.size(); i++)
        MPI_Request_free(&send_req[i]);
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double t0, tfinal;

    int n_iter = 10;
    if (num_procs > 1000)
    {
        n_iter = 100;
    }

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

    std::vector<double> sendbuf(A.send_comm.size_msgs);
    std::vector<double> sendbuf_tmp(A.send_comm.size_msgs);
    std::vector<double> recvbuf_std(A.recv_comm.size_msgs);
    std::vector<double> recvbuf_new(A.recv_comm.size_msgs);

    std::mt19937 gen(rank + time(NULL));
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::generate(sendbuf.begin(), sendbuf.end(),
        [&]() { return dist(gen); });

    MPIL_Comm* xcomm;
    MPIL_Comm_init(&xcomm, MPI_COMM_WORLD);

    MPIL_Info* xinfo;
    MPIL_Info_init(&xinfo);

    std::vector<MPI_Request> send_req;
    std::vector<MPI_Request> recv_req;
    std::vector<MPI_Request> reverse_send_req;
    std::vector<MPI_Request> reverse_recv_req;

    // Standard Communication
    comm_init(A, sendbuf, recvbuf_std, send_req, recv_req, xcomm);
    MPI_Startall(recv_req.size(), recv_req.data());
    MPI_Startall(send_req.size(), send_req.data());
    MPI_Waitall(recv_req.size(), recv_req.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(send_req.size(), send_req.data(), MPI_STATUSES_IGNORE);
    free_requests(send_req, recv_req);

    // Reorder Recvs
    comm_init(A, sendbuf, recvbuf_new, send_req, recv_req, xcomm);
    reorder_recvs(send_req, recv_req);
    MPI_Startall(recv_req.size(), recv_req.data());
    MPI_Startall(send_req.size(), send_req.data());
    MPI_Waitall(recv_req.size(), recv_req.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(send_req.size(), send_req.data(), MPI_STATUSES_IGNORE);
    free_requests(send_req, recv_req);

    // Reorder Sends and Recvs
    std::fill(recvbuf_new.begin(), recvbuf_new.end(), 0);
    comm_init(A, sendbuf, recvbuf_new, send_req, recv_req, xcomm);
    reverse_comm_init(A, recvbuf_new, sendbuf_tmp, 
                reverse_send_req, reverse_recv_req, xcomm);
    reorder_comm(send_req, recv_req, reverse_send_req, reverse_recv_req);
    free_requests(reverse_send_req, reverse_recv_req);
    MPI_Startall(recv_req.size(), recv_req.data());
    MPI_Startall(send_req.size(), send_req.data());
    MPI_Waitall(recv_req.size(), recv_req.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(send_req.size(), send_req.data(), MPI_STATUSES_IGNORE);
    free_requests(send_req, recv_req);

    MPIL_Info_free(&xinfo);
    MPIL_Comm_free(&xcomm);

    MPI_Finalize();
    return 0;
}
