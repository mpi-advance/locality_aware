#include "sparse_coll.h"
#include <cstring>
#include <vector>


int alltoall_crs_rma(int send_nnz, int* dest, int sendcount, 
        MPI_Datatype sendtype, void* sendvals,
        int* recv_nnz_ptr, int** src_ptr, int recvcount, MPI_Datatype recvtype,
        void** recvvals_ptr, MPIX_Info* xinfo, MPIX_Comm* comm)
{ 
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    // Get bytes per datatype, total bytes to be recvd (size of win_array)
    char* send_buffer;
    if (send_nnz)
        send_buffer = (char*)(sendvals);

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);
    int bytes = num_procs * recvcount * recv_bytes;

    if (comm->win_bytes != bytes
            || comm->win_type_bytes != 1)
        MPIX_Comm_win_free(comm);

    if (comm->win == MPI_WIN_NULL)
    {
        MPIX_Comm_win_init(comm, bytes, 1);
    }

    // RMA puts to find sizes recvd from each process
    memset((comm->win_array), 0, bytes);

    send_bytes *= sendcount;
    recv_bytes *= recvcount;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOPRECEDE, comm->win);
    for (int i = 0; i < send_nnz; i++)
    {
         MPI_Put(&(send_buffer[i*send_bytes]), send_bytes, MPI_CHAR,
                 dest[i], rank*recv_bytes, recv_bytes, MPI_CHAR, comm->win);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_fence(MPI_MODE_NOPUT|MPI_MODE_NOSUCCEED, comm->win);

    std::vector<int> src;
    std::vector<char> recv_buffer;
    int flag;
    for (int i = 0; i < num_procs; i++)
    {
        flag = 0;
        for (int j = 0; j < recv_bytes; j++)
        {
            if (comm->win_array[i*recv_bytes+j])
            {
                flag = 1;
                break;
            }
        }
        if (flag)
        {
            src.push_back(i);
            recv_buffer.insert(recv_buffer.end(), &(comm->win_array[i*recv_bytes]), 
                    &(comm->win_array[i*recv_bytes]) + recv_bytes);
        }
    }

    *recv_nnz_ptr = src.size();
    MPIX_Alloc((void**)src_ptr, src.size()*sizeof(int));
    MPIX_Alloc(recvvals_ptr, recv_buffer.size());
    //(*src_ptr) = (int*)MPIalloc(src.size()*sizeof(int));
    //(*recvvals_ptr) = MPIalloc(recv_buffer.size());
    memcpy((*src_ptr), src.data(), src.size()*sizeof(int));
    memcpy((*recvvals_ptr), recv_buffer.data(), recv_buffer.size());
    

    return MPI_SUCCESS;
}

int alltoall_crs_personalized(int send_nnz, int* dest, int sendcount,
        MPI_Datatype sendtype, void* sendvals,
        int* recv_nnz, int** src_ptr, int recvcount, MPI_Datatype recvtype,
        void** recvvals_ptr, MPIX_Info* xinfo, MPIX_Comm* comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    MPI_Status recv_status;
    int proc, ctr;
    int tag;
    MPIX_Comm_tag(comm, &tag);

    char* send_buffer;
    if (send_nnz)
        send_buffer = (char*)(sendvals);
    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);
    send_bytes *= sendcount;
    recv_bytes *= recvcount;

    if (!(xinfo->crs_num_initialized))
    {
        int* msg_counts = (int*)malloc(num_procs*sizeof(int));
        memset(msg_counts, 0, num_procs*sizeof(int));

        for (int i = 0; i < send_nnz; i++)
            msg_counts[dest[i]] = 1;
        MPI_Allreduce(MPI_IN_PLACE, msg_counts, num_procs, MPI_INT, MPI_SUM, comm->global_comm);
        *recv_nnz = msg_counts[rank];
        free(msg_counts);
    }

    int* src;
    char* recvvals;
    MPIX_Alloc((void**)&src, *recv_nnz*sizeof(int));
    MPIX_Alloc((void**)&recvvals, *recv_nnz*recv_bytes);
//    int* src = (int*)MPIalloc(*recv_nnz*sizeof(int));
//    void* recvvals = MPIalloc(*recv_nnz*recv_bytes);

    if (comm->n_requests < send_nnz)
        MPIX_Comm_req_resize(comm, send_nnz);

    for (int i = 0; i < send_nnz; i++)
    {
        proc = dest[i];
        MPI_Isend(&(send_buffer[i*send_bytes]), send_bytes, MPI_BYTE, proc, tag, comm->global_comm,
                &(comm->requests[i]));
    }

    ctr = 0;
    while (ctr < *recv_nnz)
    {
        char* recv_buffer = (char*)recvvals;
        MPI_Probe(MPI_ANY_SOURCE, tag, comm->global_comm, &recv_status);
        proc = recv_status.MPI_SOURCE;
        src[ctr] = proc;
        MPI_Recv(&(recv_buffer[ctr*send_bytes]), send_bytes, MPI_BYTE, proc, tag,
                comm->global_comm, &recv_status);
        ctr++;
    }

    if (send_nnz)
        MPI_Waitall(send_nnz, comm->requests, MPI_STATUSES_IGNORE);

    *src_ptr = src;
    *recvvals_ptr = recvvals;

    return MPI_SUCCESS;
}


int alltoall_crs_nonblocking(int send_nnz, int* dest, int sendcount,
        MPI_Datatype sendtype, void* sendvals,
        int* recv_nnz, int** src_ptr, int recvcount, MPI_Datatype recvtype,
        void** recvvals_ptr, MPIX_Info* xinfo, MPIX_Comm* comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    char* send_buffer;
    if (send_nnz)
        send_buffer = (char*)(sendvals);
    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);
    send_bytes *= sendcount;
    recv_bytes *= recvcount;

    int proc, ctr, flag, ibar;
    MPI_Status recv_status;
    MPI_Request bar_req;
    int tag;
    MPIX_Comm_tag(comm, &tag);

    std::vector<int> src;
    std::vector<char> recv_buffer;

    if (comm->n_requests < send_nnz)
        MPIX_Comm_req_resize(comm, send_nnz);

    for (int i = 0; i < send_nnz; i++)
    {
        proc = dest[i];
        MPI_Issend(&(send_buffer[i*send_bytes]), send_bytes, MPI_BYTE, proc, tag,
                comm->global_comm, &(comm->requests[i]));
    }

    ibar = 0;
    ctr = 0;
    while (1)
    {
        MPI_Iprobe(MPI_ANY_SOURCE, tag, comm->global_comm, &flag, &recv_status);
        if (flag)
        {
            //char* recv_buffer = (char*)recvvals;
            proc = recv_status.MPI_SOURCE;
            src.push_back(proc);
            recv_buffer.resize(src.size()*recv_bytes);
            MPI_Recv(&(recv_buffer[ctr*recv_bytes]), recv_bytes, MPI_BYTE, proc, tag,
                    comm->global_comm, &recv_status);
            ctr++;
        }
        if (ibar)
        {
           MPI_Test(&bar_req, &flag, &recv_status);
           if (flag) 
               break;
        }
        else
        {
            MPI_Testall(send_nnz, comm->requests, &flag, MPI_STATUSES_IGNORE);
            if (flag)
            {
                ibar = 1;
                MPI_Ibarrier(comm->global_comm, &bar_req);
            }
        }
    }

    *recv_nnz = src.size();
    MPIX_Alloc((void**)src_ptr, src.size()*sizeof(int));
    MPIX_Alloc(recvvals_ptr, recv_buffer.size());
    //(*src_ptr) = (int*)MPIalloc(src.size()*sizeof(int));
    //(*recvvals_ptr) = MPIalloc(recv_buffer.size());
    memcpy((*src_ptr), src.data(), src.size()*sizeof(int));
    memcpy((*recvvals_ptr), recv_buffer.data(), recv_buffer.size());


    return MPI_SUCCESS;
}

void local_redistribute(int node_recv_size, std::vector<char>& recv_buf, std::vector<int>& origins,
        int* recv_nnz, int** src_ptr, int recvcount, MPI_Datatype recvtype,
        void** recvvals_ptr, MPIX_Info* xinfo, MPIX_Comm* comm)
{
    int rank, num_procs, local_rank, PPN;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);
    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &PPN);

    int recv_bytes, int_bytes;
    MPI_Type_size(recvtype, &recv_bytes);
    MPI_Type_size(MPI_INT, &int_bytes);
    recv_bytes *= recvcount;

    int ctr, proc, idx, new_idx, count;
    MPI_Status recv_status;

    std::vector<int> msg_counts(PPN, 0);
    msg_counts.resize(PPN);
    std::fill(msg_counts.begin(), msg_counts.end(), 0);

    ctr = 0;
    while (ctr < node_recv_size)
    {
        MPI_Unpack(recv_buf.data(), node_recv_size, &ctr, &proc, 1, MPI_INT, comm->group_comm);
        proc -= (comm->rank_node * PPN);
        ctr += recv_bytes;
        msg_counts[proc] += recv_bytes + int_bytes;
    }

    std::vector<int> displs(PPN+1);
    displs[0] = 0;
    for (int i = 0; i < PPN; i++)
        displs[i+1] = displs[i] + msg_counts[i];
    
    std::vector<char> local_send_buffer;
    if (recv_buf.size())
        local_send_buffer.resize(recv_buf.size());

    ctr = 0;
    idx = 0;
    while (ctr < node_recv_size)
    {
        MPI_Unpack(recv_buf.data(), node_recv_size, &ctr, &proc, 1, MPI_INT, comm->group_comm);
        proc -= (comm->rank_node * PPN);

        MPI_Pack(&(origins[idx++]), 1, MPI_INT, local_send_buffer.data(),
                local_send_buffer.size(), &(displs[proc]), comm->local_comm);
        MPI_Pack(&(recv_buf[ctr]), recv_bytes, MPI_PACKED, local_send_buffer.data(),
                local_send_buffer.size(), &(displs[proc]), comm->local_comm);
        ctr += recv_bytes;
    }
    displs[0] = 0;
    for (int i = 0; i < PPN; i++)
        displs[i+1] = displs[i] + msg_counts[i];

    int tag;
    MPIX_Comm_tag(comm, &tag);

    MPI_Allreduce(MPI_IN_PLACE, msg_counts.data(), PPN, MPI_INT, MPI_SUM, comm->local_comm);
    int recv_count = msg_counts[local_rank];
    if (PPN > comm->n_requests)
        MPIX_Comm_req_resize(comm, PPN);

    // Send a message to every process that I will need data from
    // Tell them which global indices I need from them
    int n_sends = 0;
    for (int i = 0; i < PPN; i++)
    {
        if (displs[i+1] == displs[i])
            continue;

        MPI_Isend(&(local_send_buffer[displs[i]]), displs[i+1] - displs[i], MPI_PACKED, i, tag,
                comm->local_comm, &(comm->requests[n_sends++]));
    }

    std::vector<char> local_recv_buffer;
    ctr = recv_count * (recv_bytes + int_bytes);
    if (ctr)
        local_recv_buffer.resize(ctr);

    // Wait to receive values
    // until I have received fewer than the number of global indices I am waiting on
    ctr = 0;
    while(ctr < recv_count)
    {
        // Wait for a message
        MPI_Probe(MPI_ANY_SOURCE, tag, comm->local_comm, &recv_status);

        // Get the source process and message size
        proc = recv_status.MPI_SOURCE;
        MPI_Get_count(&recv_status, MPI_PACKED, &count);

        // Receive the message, and add local indices to send_comm
        MPI_Recv(&(local_recv_buffer[ctr]), count, MPI_PACKED, proc, tag, 
                comm->local_comm, MPI_STATUS_IGNORE);
        ctr += count;
    }
    if (n_sends) MPI_Waitall(n_sends, comm->requests, MPI_STATUSES_IGNORE);

    // Last Step : Step through recvbuf to find proc of origin, size, and indices
    std::vector<int> src;
    std::vector<char> recv_buffer;
    idx = 0;
    new_idx = 0;
    while (idx < ctr)
    {
        MPI_Unpack(local_recv_buffer.data(), local_recv_buffer.size(), &idx,
                &proc, 1, MPI_INT, comm->local_comm);
        src.push_back(proc);
        recv_buffer.resize(src.size() * recv_bytes);
        MPI_Unpack(local_recv_buffer.data(), local_recv_buffer.size(), &idx,
                &(recv_buffer[new_idx]), recv_bytes, MPI_BYTE, comm->local_comm);
        new_idx += recv_bytes;
    }
    *recv_nnz = src.size();

    MPIX_Alloc((void**)src_ptr, src.size()*sizeof(int));
    MPIX_Alloc(recvvals_ptr, recv_buffer.size());
//    (*src_ptr) = (int*)MPIalloc(src.size()*sizeof(int));
//    (*recvvals_ptr) = MPIalloc(recv_buffer.size());
    memcpy((*src_ptr), src.data(), src.size()*sizeof(int));
    memcpy((*recvvals_ptr), recv_buffer.data(), recv_buffer.size());
}

/* Assumes SMP Ordering of ranks across nodes (aggregates ranks 0-PPN) */
int alltoall_crs_personalized_loc(int send_nnz, int* dest, int sendcount, 
        MPI_Datatype sendtype, void* sendvals,
        int* recv_nnz, int** src_ptr, int recvcount, MPI_Datatype recvtype,
        void** recvvals_ptr, MPIX_Info* xinfo, MPIX_Comm* comm)
{
    int rank, num_procs, local_rank, PPN;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    if (comm->local_comm == MPI_COMM_NULL)
        MPIX_Comm_topo_init(comm);


    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &PPN);

    char* send_buffer = (char*)sendvals;
    int send_bytes, recv_bytes, int_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);
    MPI_Type_size(MPI_INT, &int_bytes);
    send_bytes *= sendcount;
    recv_bytes *= recvcount;

    if (comm->n_requests < send_nnz)
        MPIX_Comm_req_resize(comm, send_nnz);

    MPI_Status recv_status;
    int proc, ctr, start, end;
    int count, n_msgs, n_sends;
    int tag;
    MPIX_Comm_tag(comm, &tag);

    std::vector<char> node_send_buffer;
    std::vector<char> local_send_buffer;
    std::vector<char> local_recv_buffer;

    std::vector<char> recv_buffer;
    std::vector<int> src;

    count = send_nnz * (send_bytes + int_bytes);
    if (count)
        node_send_buffer.resize(count);

    // Send a message to every process that I will need data from
    // Tell them which global indices I need from them
    int group_procs, group_rank;;
    MPI_Comm_size(comm->group_comm, &group_procs);
    MPI_Comm_rank(comm->group_comm, &group_rank);

    std::vector<int> msg_counts(group_procs, 0);
    std::vector<int> msg_displs(group_procs+1);
    int node = -1;
    for (int i = 0; i < send_nnz; i++)
    {
        proc = dest[i] / PPN;
        msg_counts[proc] += int_bytes + send_bytes;
    }
    msg_displs[0] = 0;
    for (int i = 0; i < group_procs; i++)
        msg_displs[i+1] = msg_displs[i] + msg_counts[i];

    for (int i = 0; i < send_nnz; i++)
    {
        proc = dest[i];
        if (proc / PPN != node)
        {
            node = proc / PPN;
        }
        MPI_Pack(&proc, 1, MPI_INT, node_send_buffer.data(), node_send_buffer.size(), 
                &(msg_displs[node]), comm->group_comm);
        MPI_Pack(&(send_buffer[i*send_bytes]), sendcount, sendtype, node_send_buffer.data(),
                node_send_buffer.size(), &(msg_displs[node]), comm->group_comm);
    }
    msg_displs[0] = 0;
    for (int i = 0; i < group_procs; i++)
        msg_displs[i+1] = msg_displs[i] + msg_counts[i];

    MPI_Allreduce(MPI_IN_PLACE, msg_counts.data(), group_procs, MPI_INT, MPI_SUM, comm->group_comm);
    int node_recv_size = msg_counts[group_rank];

    if (send_nnz > 0)
        node = dest[0] / PPN;

    n_sends = 0;
    for (int i = 0; i < group_procs; i++)
    {
        start = msg_displs[i];
        end = msg_displs[i+1];
        if (end - start)
        {
            MPI_Isend(&(node_send_buffer[start]), end-start, MPI_PACKED, i, tag, comm->group_comm, &(comm->requests[n_sends++]));
        }
    }

    std::vector<char> recv_buf;
    if (node_recv_size)
        recv_buf.resize(node_recv_size);

    std::vector<int> origins;
    ctr = 0;
    // Wait to receive values
    // until I have received fewer than the number of global indices I am waiting on
    while (ctr < node_recv_size)
    {
        // Wait for a message
        MPI_Probe(MPI_ANY_SOURCE, tag, comm->group_comm, &recv_status);
            
        // Get the source process and message size
        proc = recv_status.MPI_SOURCE;
        MPI_Get_count(&recv_status, MPI_PACKED, &count);
        //recv_buf.resize(ctr + count);

        // Receive the message, and add local indices to send_comm
        MPI_Recv(&(recv_buf[ctr]), count, MPI_PACKED, proc, tag, comm->group_comm, 
                &recv_status);
        ctr += count;

        n_msgs = count / (recv_bytes + int_bytes);
        for (int i = 0; i < n_msgs; i++)
            origins.push_back(proc*PPN + local_rank);
    }

    local_redistribute(node_recv_size, recv_buf, origins, recv_nnz, src_ptr,
            recvcount, recvtype, recvvals_ptr, xinfo, comm);

    return MPI_SUCCESS;
}


/* Assumes SMP Ordering of ranks across nodes (aggregates ranks 0-PPN) */
int alltoall_crs_nonblocking_loc(int send_nnz, int* dest, int sendcount, 
        MPI_Datatype sendtype, void* sendvals,
        int* recv_nnz, int** src_ptr, int recvcount, MPI_Datatype recvtype,
        void** recvvals_ptr, MPIX_Info* xinfo, MPIX_Comm* comm)
{ 
    int rank, num_procs, local_rank, PPN;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    if (comm->local_comm == MPI_COMM_NULL)
        MPIX_Comm_topo_init(comm);

    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &PPN);

    char* send_buffer = (char*)sendvals;
    int send_bytes, recv_bytes, int_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);
    MPI_Type_size(MPI_INT, &int_bytes);
    send_bytes *= sendcount;
    recv_bytes *= recvcount;

    if (comm->n_requests < send_nnz)
        MPIX_Comm_req_resize(comm, send_nnz);

    MPI_Status recv_status;
    MPI_Request bar_req;
    int proc, ctr, flag, ibar, start, end;
    int count, n_msgs, n_sends;
    int tag;
    MPIX_Comm_tag(comm, &tag);

    std::vector<char> node_send_buffer;
    std::vector<char> local_send_buffer;
    std::vector<char> local_recv_buffer;

    count = send_nnz * (send_bytes + int_bytes);
    if (count)
        node_send_buffer.resize(count);

    int group_procs, group_rank;;
    MPI_Comm_size(comm->group_comm, &group_procs);
    MPI_Comm_rank(comm->group_comm, &group_rank);

    // Send a message to every process that I will need data from
    // Tell them which global indices I need from them
    std::vector<int> msg_counts(group_procs, 0);
    std::vector<int> msg_displs(group_procs+1);
    int node = -1;
    for (int i = 0; i < send_nnz; i++)
    {
        proc = dest[i] / PPN;
        msg_counts[proc] += int_bytes + send_bytes;
    }
    msg_displs[0] = 0;
    for (int i = 0; i < group_procs; i++)
        msg_displs[i+1] = msg_displs[i] + msg_counts[i];

    for (int i = 0; i < send_nnz; i++)
    {
        proc = dest[i];
        if (proc / PPN != node)
        {
            node = proc / PPN;
        }
        MPI_Pack(&proc, 1, MPI_INT, node_send_buffer.data(), node_send_buffer.size(), 
                &(msg_displs[node]), comm->group_comm);
        MPI_Pack(&(send_buffer[i*send_bytes]), sendcount, sendtype, node_send_buffer.data(),
                node_send_buffer.size(), &(msg_displs[node]), comm->group_comm);
    }
    msg_displs[0] = 0;
    for (int i = 0; i < group_procs; i++)
        msg_displs[i+1] = msg_displs[i] + msg_counts[i];

    if (send_nnz > 0)
        node = dest[0] / PPN;

    n_sends = 0;
    for (int i = 0; i < group_procs; i++)
    {
        start = msg_displs[i];
        end = msg_displs[i+1];
        if (end - start)
        {
            MPI_Issend(&(node_send_buffer[start]), end-start, MPI_PACKED, i, tag, comm->group_comm, &(comm->requests[n_sends++]));
        }
    }


    std::vector<char> recv_buf;
    std::vector<int> origins;
    ibar = 0;
    ctr = 0;
    // Wait to receive values
    // until I have received fewer than the number of global indices I am waiting on
    while (1)
    {
        // Wait for a message
        MPI_Iprobe(MPI_ANY_SOURCE, tag, comm->group_comm, &flag, &recv_status);
        if (flag)
        {
            // Get the source process and message size
            proc = recv_status.MPI_SOURCE;
            MPI_Get_count(&recv_status, MPI_PACKED, &count);
            recv_buf.resize(ctr + count);

            // Receive the message, and add local indices to send_comm
            MPI_Recv(&(recv_buf[ctr]), count, MPI_PACKED, proc, tag, comm->group_comm, 
                    &recv_status);
            ctr += count;


            n_msgs = count / (recv_bytes + int_bytes);
            for (int i = 0; i < n_msgs; i++)
                origins.push_back(proc*PPN+local_rank);

        }


        // If I have already called my Ibarrier, check if all processes have reached
        // If all processes have reached the Ibarrier, all messages have been sent
        if (ibar)
        {
            MPI_Test(&bar_req, &flag, &recv_status);
            if (flag) break;
        }
        else
        {
            // Test if all of my synchronous sends have completed.
            // They only complete once actually received.
            MPI_Testall(n_sends, comm->requests, &flag, MPI_STATUSES_IGNORE);
            if (flag)
            {
                ibar = 1;
                MPI_Ibarrier(comm->group_comm, &bar_req);
            }
        }
    }

    int node_recv_size = recv_buf.size();
    local_redistribute(node_recv_size, recv_buf, origins, recv_nnz, src_ptr,
            recvcount, recvtype, recvvals_ptr, xinfo, comm);

    return MPI_SUCCESS;
}




