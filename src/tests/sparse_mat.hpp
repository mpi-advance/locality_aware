#ifndef MPI_SPARSE_MAT_HPP
#define MPI_SPARSE_MAT_HPP

#include "mpi.h"
#include <vector>

class Mat
{
  public:
    std::vector<int> rowptr;
    std::vector<int> col_idx;
    std::vector<double> data;
    int n_rows;
    int n_cols;
    int nnz;
};

template <typename U>
class Comm 
{
  public:
    int n_msgs;
    int size_msgs;
    std::vector<int> procs;
    std::vector<U> ptr;
    std::vector<int> counts;
    std::vector<int> idx;
    std::vector<MPI_Request> req;

    Comm()
    {
        n_msgs = 0;
        size_msgs = 0;
    }

    ~Comm()
    {
    }
};

template <typename U>
class ParMat 
{
  public:
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
    Comm<U>* send_comm;
    Comm<U>* recv_comm;
    MPI_Comm dist_graph_comm;

    ParMat()
    {
        send_comm = new Comm<U>();
	recv_comm = new Comm<U>();
    }

    void reset_comm()
    {
	delete send_comm;
	send_comm = new Comm<U>();
    }
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
            A.recv_comm->procs.push_back(proc);
            A.recv_comm->ptr.push_back((U)(i));
            prev_proc = proc;
        }
    }

    // Set Recv Sizes
    A.recv_comm->ptr.push_back((U)(A.off_proc_num_cols));
    A.recv_comm->n_msgs = A.recv_comm->procs.size();
    A.recv_comm->size_msgs = A.off_proc_num_cols;
    if (A.recv_comm->n_msgs == 0)
        return;

    A.recv_comm->req.resize(A.recv_comm->n_msgs);
    A.recv_comm->counts.resize(A.recv_comm->n_msgs);
    for (int i = 0; i < A.recv_comm->n_msgs; i++)
        A.recv_comm->counts[i] = A.recv_comm->ptr[i+1] - A.recv_comm->ptr[i];
}

// Must Form Recv Comm before Send!
template <typename U>
void form_send_comm_standard(ParMat<U>& A)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::vector<long> recv_buf;
    std::vector<int> sizes(num_procs, 0);
    int start, end, proc, count, ctr;
    MPI_Status recv_status;

    // Allreduce to find size of data I will receive
    for (int i = 0; i < A.recv_comm->n_msgs; i++)
        sizes[A.recv_comm->procs[i]] = A.recv_comm->ptr[i+1] - A.recv_comm->ptr[i];
    MPI_Allreduce(MPI_IN_PLACE, sizes.data(), num_procs, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    A.send_comm->size_msgs = sizes[rank];

    // Send a message to every process that I will need data from
    // Tell them which global indices I need from them
    int msg_tag = 1234;
    for (int i = 0; i < A.recv_comm->n_msgs; i++)
    {
        proc = A.recv_comm->procs[i];
        MPI_Isend(&(A.off_proc_columns[A.recv_comm->ptr[i]]), A.recv_comm->counts[i], MPI_LONG, proc, msg_tag, 
                MPI_COMM_WORLD, &(A.recv_comm->req[i]));
    }

    // Wait to receive values
    // until I have received fewer than the number of global indices I am waiting on
    if (A.send_comm->size_msgs)
    {
        A.send_comm->idx.resize(A.send_comm->size_msgs);
        recv_buf.resize(A.send_comm->size_msgs);
    }
    ctr = 0;
    A.send_comm->ptr.push_back(0);
    while (ctr < A.send_comm->size_msgs)
    {
        // Wait for a message
        MPI_Probe(MPI_ANY_SOURCE, msg_tag, MPI_COMM_WORLD, &recv_status);

        // Get the source process and message size
        proc = recv_status.MPI_SOURCE;
        A.send_comm->procs.push_back(proc);
        MPI_Get_count(&recv_status, MPI_LONG, &count);
        A.send_comm->counts.push_back(count);

        // Receive the message, and add local indices to send_comm
        MPI_Recv(&(recv_buf[ctr]), count, MPI_LONG, proc, msg_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < count; i++)
        {
            A.send_comm->idx[ctr+i] = (recv_buf[ctr+i] - A.first_col);
        }
        ctr += count;
        A.send_comm->ptr.push_back((U)(ctr));
    }
    
    // Set send sizes
    A.send_comm->n_msgs = A.send_comm->procs.size();

    if (A.send_comm->n_msgs)
        A.send_comm->req.resize(A.send_comm->n_msgs);

    if (A.recv_comm->n_msgs)
        MPI_Waitall(A.recv_comm->n_msgs, A.recv_comm->req.data(), MPI_STATUSES_IGNORE);
}

// Must Form Recv Comm before Send!
template <typename U>
void form_send_comm_torsten(ParMat<U>& A)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::vector<long> recv_buf;
    int start, end, proc, count, ctr, flag;
    int ibar = 0;
    MPI_Status recv_status;
    MPI_Request bar_req;

    // Allreduce to find size of data I will receive

    // Send a message to every process that I will need data from
    // Tell them which global indices I need from them
    int msg_tag = 1234;
    for (int i = 0; i < A.recv_comm->n_msgs; i++)
    {
        proc = A.recv_comm->procs[i];
        MPI_Issend(&(A.off_proc_columns[A.recv_comm->ptr[i]]), A.recv_comm->counts[i], MPI_LONG, proc, msg_tag, 
                MPI_COMM_WORLD, &(A.recv_comm->req[i]));
    }

    // Wait to receive values
    // until I have received fewer than the number of global indices I am waiting on
    ctr = 0;
    A.send_comm->ptr.push_back(0);
    while (1)
    {
        // Wait for a message
        MPI_Iprobe(MPI_ANY_SOURCE, msg_tag, MPI_COMM_WORLD, &flag, &recv_status);
        if (flag)
        {
            // Get the source process and message size
            proc = recv_status.MPI_SOURCE;
            A.send_comm->procs.push_back(proc);
            MPI_Get_count(&recv_status, MPI_LONG, &count);
            A.send_comm->counts.push_back(count);
            if (count > recv_buf.size()) recv_buf.resize(count);

            // Receive the message, and add local indices to send_comm
            MPI_Recv(recv_buf.data(), count, MPI_LONG, proc, msg_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < count; i++)
            {
                A.send_comm->idx.push_back(recv_buf[i] - A.first_col);
            }
            ctr += count;
            A.send_comm->ptr.push_back((U)(ctr));
        }
        
        // If I have already called my Ibarrier, check if all processes have reached
        // If all processes have reached the Ibarrier, all messages have been sent
        if (ibar)
        {
            MPI_Test(&bar_req, &flag, MPI_STATUS_IGNORE);
            if (flag) break;
        }
        else
        {
            // Test if all of my synchronous sends have completed.
            // They only complete once actually received.
            MPI_Testall(A.recv_comm->n_msgs, A.recv_comm->req.data(), &flag, MPI_STATUSES_IGNORE);
            if (flag)
            {
                ibar = 1;
                MPI_Ibarrier(MPI_COMM_WORLD, &bar_req);
            }    
        }
    }
    
    // Set send sizes
    A.send_comm->n_msgs = A.send_comm->procs.size();
    A.send_comm->size_msgs = ctr;
    if (A.send_comm->n_msgs)
        A.send_comm->req.resize(A.send_comm->n_msgs);
    if (A.send_comm->size_msgs)
        A.send_comm->idx.resize(A.send_comm->size_msgs);
}

// Must Form Recv Comm before Send!
template <typename U>
void form_send_comm_rma(ParMat<U>& A)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::vector<long> recv_buf;
    int start, end, proc, count, ctr;
    MPI_Status recv_status;
    int bytes;

    // RMA puts to find sizes recvd from each process
    MPI_Win win;
    int* sizes;
    MPI_Alloc_mem(num_procs*sizeof(int), MPI_INFO_NULL, &sizes);
    for (int i = 0; i < num_procs; i++)
        sizes[i] = 0;
    MPI_Win_create(sizes, num_procs*sizeof(int), sizeof(int),
            MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOPRECEDE, win);
    for (int i = 0; i < A.recv_comm->n_msgs; i++)
    {
        MPI_Put(&(A.recv_comm->counts[i]), 1, MPI_INT, A.recv_comm->procs[i], 
                rank, 1, MPI_INT, win);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_fence(MPI_MODE_NOPUT|MPI_MODE_NOSUCCEED, win);
    MPI_Win_free(&win);

    A.send_comm->ptr.push_back(0);
    ctr = 0;
    for (int i = 0; i < num_procs; i++)
    {
        if (sizes[i])
        {
            A.send_comm->procs.push_back(i);
            A.send_comm->counts.push_back(sizes[i]);
            A.send_comm->ptr.push_back(A.send_comm->ptr[ctr] + sizes[i]);
            ctr++;
        }
    }

    A.send_comm->n_msgs = ctr;
    if (A.send_comm->n_msgs)
        A.send_comm->req.resize(A.send_comm->n_msgs);
    A.send_comm->size_msgs = A.send_comm->ptr[A.send_comm->n_msgs];
    if (A.send_comm->size_msgs)
    {
        A.send_comm->idx.resize(A.send_comm->size_msgs);
        recv_buf.resize(A.send_comm->size_msgs);
    }

    MPI_Free_mem(sizes);

    int msg_tag = 1234;
    for (int i = 0; i < A.send_comm->n_msgs; i++)
    {
        MPI_Irecv(&(recv_buf[A.send_comm->ptr[i]]), A.send_comm->counts[i], MPI_LONG, 
                A.send_comm->procs[i], msg_tag, MPI_COMM_WORLD, &(A.send_comm->req[i]));
    }
    for (int i = 0; i < A.recv_comm->n_msgs; i++)
    {
        MPI_Isend(&(A.off_proc_columns[A.recv_comm->ptr[i]]), A.recv_comm->counts[i], MPI_LONG, 
                A.recv_comm->procs[i], msg_tag, MPI_COMM_WORLD, &(A.recv_comm->req[i]));
    }

    if (A.send_comm->n_msgs)
        MPI_Waitall(A.send_comm->n_msgs, A.send_comm->req.data(), MPI_STATUSES_IGNORE);

    for (int i = 0; i < A.send_comm->size_msgs; i++)
        A.send_comm->idx[i] = recv_buf[i] - A.first_col;

    if (A.recv_comm->n_msgs)
        MPI_Waitall(A.recv_comm->n_msgs, A.recv_comm->req.data(), MPI_STATUSES_IGNORE);
}


// Must Form Recv Comm before Send!
void allocate_rma_dynamic(MPI_Win* win, int** sizes)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    MPI_Alloc_mem(num_procs*sizeof(int), MPI_INFO_NULL, sizes);
    MPI_Win_create(*sizes, num_procs*sizeof(int), sizeof(int),
            MPI_INFO_NULL, MPI_COMM_WORLD, win);

}

void free_rma_dynamic(MPI_Win* win, int* sizes)
{
    MPI_Win_free(win);
    MPI_Free_mem(sizes);
}

template <typename U>
void form_send_comm_rma_dynamic(ParMat<U>& A, MPI_Win win, int* sizes)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::vector<long> recv_buf;
    int start, end, proc, count, ctr;
    MPI_Status recv_status;
    int bytes;

    for (int i = 0; i < num_procs; i++)
        sizes[i] = 0;

    // RMA puts to find sizes recvd from each process
    MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOPRECEDE, win);

    // puts #1
    for (int i = 0; i < A.recv_comm->n_msgs; i++)
    {
        MPI_Put(&(A.recv_comm->counts[i]), 1, MPI_INT, A.recv_comm->procs[i],
               rank, 1, MPI_INT, win);
    }

    MPI_Win_fence(MPI_MODE_NOPUT|MPI_MODE_NOSUCCEED, win);

    A.send_comm->ptr.push_back(0);
    ctr = 0;
    for (int i = 0; i < num_procs; i++)
    {
        if (sizes[i])
        {
            A.send_comm->procs.push_back(i);
            A.send_comm->counts.push_back(sizes[i]);
            A.send_comm->ptr.push_back(A.send_comm->ptr[ctr] + sizes[i]);
            ctr++;
        }
    }

    A.send_comm->n_msgs = ctr;
    if (A.send_comm->n_msgs)
        A.send_comm->req.resize(A.send_comm->n_msgs);
    A.send_comm->size_msgs = A.send_comm->ptr[A.send_comm->n_msgs];
    if (A.send_comm->size_msgs)
    {
        A.send_comm->idx.resize(A.send_comm->size_msgs);
        recv_buf.resize(A.send_comm->size_msgs);
    }

    int msg_tag = 1234;
    for (int i = 0; i < A.send_comm->n_msgs; i++)
    {
        MPI_Irecv(&(recv_buf[A.send_comm->ptr[i]]), A.send_comm->counts[i], MPI_LONG,
                A.send_comm->procs[i], msg_tag, MPI_COMM_WORLD, &(A.send_comm->req[i]));
    }
    for (int i = 0; i < A.recv_comm->n_msgs; i++)
    {
        MPI_Isend(&(A.off_proc_columns[A.recv_comm->ptr[i]]), A.recv_comm->counts[i], MPI_LONG,
                A.recv_comm->procs[i], msg_tag, MPI_COMM_WORLD, &(A.recv_comm->req[i]));
    }

    if (A.send_comm->n_msgs)
        MPI_Waitall(A.send_comm->n_msgs, A.send_comm->req.data(), MPI_STATUSES_IGNORE);

    for (int i = 0; i < A.send_comm->size_msgs; i++)
        A.send_comm->idx[i] = recv_buf[i] - A.first_col;

    if (A.recv_comm->n_msgs)
        MPI_Waitall(A.recv_comm->n_msgs, A.recv_comm->req.data(), MPI_STATUSES_IGNORE);
}


template <typename U>
void form_comm(ParMat<U>& A)
{
    // Form Recv Side 
    form_recv_comm(A);

    // Form Send Side (Algorithm Options Here!)
    //form_send_comm_standard(A);
    //form_send_comm_torsten(A);
    form_send_comm_rma(A);
}


template <typename U, typename T>
void communicate(ParMat<T>& A, std::vector<U>& data, std::vector<U>& recvbuf, MPI_Datatype type)
{
    int proc;
    T start, end;
    int tag = 2948;
    std::vector<U> sendbuf;
    if (A.send_comm->size_msgs)
        sendbuf.resize(A.send_comm->size_msgs);
    for (int i = 0; i < A.send_comm->n_msgs; i++)
    {
        proc = A.send_comm->procs[i];
        start = A.send_comm->ptr[i];
        end = A.send_comm->ptr[i+1];
        for (T j = start; j < end; j++)
        {
            sendbuf[j] = data[A.send_comm->idx[j]];
        }
        MPI_Isend(&(sendbuf[start]), (int)(end - start), type, proc, tag, 
                MPI_COMM_WORLD, &(A.send_comm->req[i]));
    }

    for (int i = 0; i < A.recv_comm->n_msgs; i++)
    {
        proc = A.recv_comm->procs[i];
        start = A.recv_comm->ptr[i];
        end = A.recv_comm->ptr[i+1];
        MPI_Irecv(&(recvbuf[start]), (int)(end - start), type, proc, tag,
                MPI_COMM_WORLD, &(A.recv_comm->req[i]));
    }

    if (A.send_comm->n_msgs)
        MPI_Waitall(A.send_comm->n_msgs, A.send_comm->req.data(), MPI_STATUSES_IGNORE);
    if (A.recv_comm->n_msgs)
    MPI_Waitall(A.recv_comm->n_msgs, A.recv_comm->req.data(), MPI_STATUSES_IGNORE);
}

#endif

