#ifndef MPI_SPARSE_MAT_HPP
#define MPI_SPARSE_MAT_HPP

#include "mpi.h"
#include <vector>

enum COMM_ALGORITHM { STANDARD, TORSTEN, RMA, RMA_DYNAMIC };

struct Mat 
{
    std::vector<int> rowptr;
    std::vector<int> col_idx;
    std::vector<double> data;
    int n_rows;
    int n_cols;
    int nnz;
};


template <typename U>
struct Comm
{
    int n_msgs;
    int size_msgs;
    std::vector<int> procs;
    std::vector<U> ptr;
    std::vector<int> counts;
    std::vector<int> idx;
    std::vector<MPI_Request> req;
};

template <typename U>
struct ParMat
{
    Mat on_proc;
    Mat off_proc;
    int global_rows;
    int global_cols;
    int local_rows;
    int local_cols;
    int first_row;
    int first_col;
    int off_proc_num_cols;
    std::vector<long> off_proc_columns;
    Comm<U> send_comm;
    Comm<U> recv_comm;
    MPI_Comm dist_graph_comm;
};

template <typename U>
void form_recv_comm(ParMat<U>& A)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Gather first col for all processes into list
    std::vector<int> first_cols(num_procs+1);
    MPI_Allgather(&A.first_col, 1, MPI_INT, first_cols.data(), 1, MPI_INT, MPI_COMM_WORLD);
    first_cols[num_procs] = A.global_cols;

    // Map Columns to Processes
    int proc = 0;
    int prev_proc = -1;
    for (int i = 0; i < A.off_proc_num_cols; i++)
    {
        int global_col = A.off_proc_columns[i];
        while (first_cols[proc+1] <= global_col)
            proc++;
        if (proc != prev_proc)
        {
            A.recv_comm.procs.push_back(proc);
            A.recv_comm.ptr.push_back((U)(i));
            prev_proc = proc;
        }
    }

    // Set Recv Sizes
    A.recv_comm.ptr.push_back((U)(A.off_proc_num_cols));
    A.recv_comm.n_msgs = A.recv_comm.procs.size();
    A.recv_comm.size_msgs = A.off_proc_num_cols;
    if (A.recv_comm.n_msgs == 0)
        return;

    A.recv_comm.req.resize(A.recv_comm.n_msgs);
    A.recv_comm.counts.resize(A.recv_comm.n_msgs);
    for (int i = 0; i < A.recv_comm.n_msgs; i++)
        A.recv_comm.counts[i] = A.recv_comm.ptr[i+1] - A.recv_comm.ptr[i];
}

template <typename U, typename T>
void communicate(ParMat<T>& A, std::vector<U>& data, std::vector<U>& recvbuf, MPI_Datatype type)
{
    int proc;
    T start, end;
    int tag = 2948;
    std::vector<U> sendbuf;
    if (A.send_comm.size_msgs)
        sendbuf.resize(A.send_comm.size_msgs);
    for (int i = 0; i < A.send_comm.n_msgs; i++)
    {
        proc = A.send_comm.procs[i];
        start = A.send_comm.ptr[i];
        end = A.send_comm.ptr[i+1];
        for (T j = start; j < end; j++)
        {
            sendbuf[j] = data[A.send_comm.idx[j]];
        }
        MPI_Isend(&(sendbuf[start]), (int)(end - start), type, proc, tag, 
                MPI_COMM_WORLD, &(A.send_comm.req[i]));
    }

    for (int i = 0; i < A.recv_comm.n_msgs; i++)
    {
        proc = A.recv_comm.procs[i];
        start = A.recv_comm.ptr[i];
        end = A.recv_comm.ptr[i+1];
        MPI_Irecv(&(recvbuf[start]), (int)(end - start), type, proc, tag,
                MPI_COMM_WORLD, &(A.recv_comm.req[i]));
    }

    if (A.send_comm.n_msgs)
        MPI_Waitall(A.send_comm.n_msgs, A.send_comm.req.data(), MPI_STATUSES_IGNORE);
    if (A.recv_comm.n_msgs)
    MPI_Waitall(A.recv_comm.n_msgs, A.recv_comm.req.data(), MPI_STATUSES_IGNORE);
}
#endif
