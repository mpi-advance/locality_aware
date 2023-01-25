#ifndef MPI_SPARSE_MAT_HPP
#define MPI_SPARSE_MAT_HPP

#include "mpi.h"
#include <vector>

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
void form_comm(ParMat<U>& A)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Gather first row for all processes into list
    std::vector<int> first_rows(num_procs+1);
    MPI_Allgather(&A.first_row, 1, MPI_INT, first_rows.data(), 1, MPI_INT, MPI_COMM_WORLD);
    first_rows[num_procs] = A.global_rows;

    // Step through off_proc_columns and find which process the corresponding row is stored on
    std::vector<int> col_to_proc(A.off_proc_num_cols);
    int proc = 0;
    int prev_proc = -1;
    std::vector<int> sizes(num_procs);

    for (int i = 0; i < A.off_proc_num_cols; i++)
    {
        int global_col = A.off_proc_columns[i];
        while (first_rows[proc+1] <= global_col)
            proc++;
        col_to_proc[i] = proc;
        if (proc != prev_proc)
        {
            A.recv_comm.procs.push_back(proc);
            A.recv_comm.ptr.push_back((U)(i));
            prev_proc = proc;
            sizes[proc] = 1;
        }
    }
    A.recv_comm.ptr.push_back((U)(A.off_proc_num_cols));
    A.recv_comm.n_msgs = A.recv_comm.procs.size();
    A.recv_comm.req.resize(A.recv_comm.n_msgs);
    A.recv_comm.size_msgs = A.off_proc_num_cols;

    // Reduce NSends to Each Proc
    MPI_Allreduce(MPI_IN_PLACE, sizes.data(), num_procs, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    A.send_comm.n_msgs = sizes[rank];

    int msg_tag = 1234;
    A.recv_comm.counts.resize(A.recv_comm.n_msgs);
    for (int i = 0; i < A.recv_comm.n_msgs; i++)
    {
        proc = A.recv_comm.procs[i];
        U start = A.recv_comm.ptr[i];
        U end = A.recv_comm.ptr[i+1];
        A.recv_comm.counts[i] = (int)(end - start);
        MPI_Isend(&(A.off_proc_columns[start]), A.recv_comm.counts[i], MPI_LONG, proc, msg_tag, 
                MPI_COMM_WORLD, &(A.recv_comm.req[i]));
    }

    MPI_Status recv_status;
    std::vector<long> recv_buf;
    int count_sum = 0;
    int count;
    A.send_comm.ptr.push_back(0);
    for (int i = 0; i < A.send_comm.n_msgs; i++)
    {
        MPI_Probe(MPI_ANY_SOURCE, msg_tag, MPI_COMM_WORLD, &recv_status);
        proc = recv_status.MPI_SOURCE;
        A.send_comm.procs.push_back(proc);
        MPI_Get_count(&recv_status, MPI_LONG, &count);
        A.send_comm.counts.push_back(count);
        count_sum += count;
        A.send_comm.ptr.push_back((U)(count_sum));
        if (recv_buf.size() < count) recv_buf.resize(count);
        MPI_Recv(recv_buf.data(), count, MPI_LONG, proc, msg_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < count; i++)
        {
            A.send_comm.idx.push_back(recv_buf[i] - A.first_row);
        }
    }
    A.send_comm.req.resize(A.send_comm.n_msgs);
    A.send_comm.size_msgs = count_sum;

    MPI_Waitall(A.recv_comm.n_msgs, A.recv_comm.req.data(), MPI_STATUSES_IGNORE);
}


template <typename U, typename T>
void communicate(ParMat<T>& A, std::vector<U>& data, std::vector<U>& recvbuf, MPI_Datatype type)
{
    int proc;
    T start, end;
    int tag = 2948;
    std::vector<U> sendbuf(A.send_comm.idx.size());
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

    MPI_Waitall(A.send_comm.n_msgs, A.send_comm.req.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(A.recv_comm.n_msgs, A.recv_comm.req.data(), MPI_STATUSES_IGNORE);
}



#endif
