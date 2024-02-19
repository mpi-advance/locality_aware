#include "sparse_coll.h"
#include <cstring>
#include <vector>

/* Assumes SMP Ordering of ranks across nodes (aggregates ranks 0-PPN) */
int alltoall_crs_personalized_loc(int send_nnz, int* dest, int sendcount, 
        MPI_Datatype sendtype, void* sendvals,
        int* recv_nnz, int* src, int recvcount, MPI_Datatype recvtype,
        void* recvvals, MPIX_Info* xinfo, MPIX_Comm* comm)
{
    int rank, num_procs, local_rank, PPN;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    if (comm->local_comm == MPI_COMM_NULL)
        MPIX_Comm_topo_init(comm);

    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &PPN);

    char* send_buffer = (char*)sendvals;
    char* recv_buffer = (char*)recvvals;
    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);
    send_bytes *= sendcount;
    recv_bytes *= recvcount;

    if (comm->n_requests < send_nnz)
        MPIX_Comm_req_resize(comm, send_nnz);

    MPI_Status recv_status;
    MPI_Request bar_req;
    int proc, ctr, flag, ibar;
    int first, last, count, n_msgs, n_sends, n_recvs, idx, new_idx;
    int tag = xinfo->tag;
    xinfo->tag = (xinfo->tag + 1 % MPI_TAG_UB);

    std::vector<char> node_send_buffer;
    std::vector<char> local_send_buffer;
    std::vector<char> local_recv_buffer;

    count = send_nnz * (send_bytes + sizeof(int));
    if (count)
        node_send_buffer.resize(count);

    // Send a message to every process that I will need data from
    // Tell them which global indices I need from them
    std::vector<int> msg_counts(num_procs, 0);
    int node = -1;
    for (int i = 0; i < send_nnz; i++)
    {
        proc = (dest[i] / PPN) * PPN + local_rank;
        msg_counts[proc] += sizeof(int) + send_bytes;
    }
    MPI_Allreduce(MPI_IN_PLACE, msg_counts.data(), num_procs, MPI_INT, MPI_SUM, comm->global_comm);
    int node_recv_size = msg_counts[rank];

    if (send_nnz > 0)
        node = dest[0] / PPN;

    first = 0;
    last = 0;
    n_sends = 0;
    for (int i = 0; i < send_nnz; i++)
    {
        proc = dest[i];
        if (proc / PPN != node)
        {
            MPI_Isend(&(node_send_buffer[first]), (last - first), MPI_BYTE,
                    node*PPN + local_rank, tag, comm->global_comm, &(comm->requests[n_sends++]));
            first = last;
            node = proc / PPN;
        }
        memcpy(&(node_send_buffer[last]), &proc, sizeof(int));
        last += sizeof(int);
        memcpy(&(node_send_buffer[last]), &(send_buffer[i*send_bytes]), send_bytes);
        last += send_bytes;
    }
    if (node >= 0)
    {
        MPI_Isend(&(node_send_buffer[first]), (last - first), MPI_BYTE,
                node*PPN + local_rank, tag, comm->global_comm, &(comm->requests[n_sends++]));
    }


    std::vector<char> recv_buf;
    if (node_recv_size)
        recv_buf.resize(node_recv_size);

    std::vector<int> origins;
    ibar = 0;
    ctr = 0;
    // Wait to receive values
    // until I have received fewer than the number of global indices I am waiting on
    while (ctr < node_recv_size)
    {
        // Wait for a message
        MPI_Probe(MPI_ANY_SOURCE, tag, comm->global_comm, &recv_status);
            
        // Get the source process and message size
        proc = recv_status.MPI_SOURCE;
        MPI_Get_count(&recv_status, MPI_BYTE, &count);
        //recv_buf.resize(ctr + count);

        // Receive the message, and add local indices to send_comm
        MPI_Recv(&(recv_buf[ctr]), count, MPI_BYTE, proc, tag, comm->global_comm, 
                &recv_status);
        ctr += count;

        n_msgs = count / (recv_bytes + sizeof(int));
        for (int i = 0; i < n_msgs; i++)
            origins.push_back(proc);
    }

    msg_counts.resize(PPN);
    std::fill(msg_counts.begin(), msg_counts.end(), 0);

    ctr = 0;
    while (ctr < recv_buf.size())
    {
        memcpy(&proc, &(recv_buf[ctr]), sizeof(int));
        proc -= (comm->rank_node * PPN);
        ctr += recv_bytes + sizeof(int);
        msg_counts[proc] += recv_bytes + sizeof(int);
    }

    std::vector<int> displs(PPN+1);
    displs[0] = 0;
    for (int i = 0; i < PPN; i++)
        displs[i+1] = displs[i] + msg_counts[i];
    
    count = recv_buf.size();
    if (count)
        local_send_buffer.resize(count);

    ctr = 0;
    idx = 0;
    while (ctr < recv_buf.size())
    {
        memcpy(&proc, &(recv_buf[ctr]), sizeof(int));
        proc -= (comm->rank_node * PPN);

        memcpy(&(local_send_buffer[displs[proc]]), &(origins[idx]), sizeof(int));
        ctr += sizeof(int);
        displs[proc] += sizeof(int);

        memcpy(&(local_send_buffer[displs[proc]]), &(recv_buf[ctr]), recv_bytes);
        ctr += recv_bytes;
        displs[proc] += recv_bytes;
        idx++;
    }
    displs[0] = 0;
    for (int i = 0; i < PPN; i++)
        displs[i+1] = displs[i] + msg_counts[i];

    tag = xinfo->tag;
    xinfo->tag = (xinfo->tag + 1 % MPI_TAG_UB);

    MPI_Allreduce(MPI_IN_PLACE, msg_counts.data(), PPN, MPI_INT, MPI_SUM, comm->local_comm);
    int recv_count = msg_counts[local_rank];
    if (PPN > comm->n_requests)
        MPIX_Comm_req_resize(comm, PPN);

    // Send a message to every process that I will need data from
    // Tell them which global indices I need from them
    n_sends = 0;
    for (int i = 0; i < PPN; i++)
    {
        if (displs[i+1] == displs[i])
            continue;

        MPI_Isend(&(local_send_buffer[displs[i]]), displs[i+1] - displs[i], MPI_BYTE, i, tag,
                comm->local_comm, &(comm->requests[n_sends++]));
    }

    count = recv_count * (recv_bytes + sizeof(int));
    if (count)
        local_recv_buffer.resize(count);

    // Wait to receive values
    // until I have received fewer than the number of global indices I am waiting on
    ctr = 0;
    while(ctr < recv_count)
    {
        // Wait for a message
        MPI_Probe(MPI_ANY_SOURCE, tag, comm->local_comm, &recv_status);

        // Get the source process and message size
        proc = recv_status.MPI_SOURCE;
        MPI_Get_count(&recv_status, MPI_BYTE, &count);

        // Receive the message, and add local indices to send_comm
        MPI_Recv(&(local_recv_buffer[ctr]), count, MPI_BYTE, proc, tag, 
                comm->local_comm, MPI_STATUS_IGNORE);
        ctr += count;
    }
    if (n_sends) MPI_Waitall(n_sends, comm->requests, MPI_STATUSES_IGNORE);

    // Last Step : Step through recvbuf to find proc of origin, size, and indices
    idx = 0;
    new_idx = 0;
    n_recvs = 0;
    while (idx < ctr)
    {
        memcpy(&proc, &(local_recv_buffer[idx]), sizeof(int));
        src[n_recvs++] = proc;
        idx += sizeof(int);
        memcpy(&(recv_buffer[new_idx]), &(local_recv_buffer[idx]), recv_bytes);
        idx += recv_bytes;
        new_idx += recv_bytes;
    }
    *recv_nnz = n_recvs;

    return MPI_SUCCESS;
}



/* Assumes SMP Ordering of ranks across nodes (aggregates ranks 0-PPN) */
int alltoall_crs_nonblocking_loc(int send_nnz, int* dest, int sendcount, 
        MPI_Datatype sendtype, void* sendvals,
        int* recv_nnz, int* src, int recvcount, MPI_Datatype recvtype,
        void* recvvals, MPIX_Info* xinfo, MPIX_Comm* comm)
{ 
    int rank, num_procs, local_rank, PPN;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    if (comm->local_comm == MPI_COMM_NULL)
        MPIX_Comm_topo_init(comm);

    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &PPN);

    char* send_buffer = (char*)sendvals;
    char* recv_buffer = (char*)recvvals;
    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);
    send_bytes *= sendcount;
    recv_bytes *= recvcount;

    if (comm->n_requests < send_nnz)
        MPIX_Comm_req_resize(comm, send_nnz);

    MPI_Status recv_status;
    MPI_Request bar_req;
    int proc, ctr, flag, ibar;
    int first, last, count, n_msgs, n_sends, n_recvs, idx, new_idx;
    int tag = xinfo->tag;
    xinfo->tag = (xinfo->tag + 1 % MPI_TAG_UB);

    std::vector<char> node_send_buffer;
    std::vector<char> local_send_buffer;
    std::vector<char> local_recv_buffer;

    count = send_nnz * (send_bytes + sizeof(int));
    if (count)
        node_send_buffer.resize(count);

    // Send a message to every process that I will need data from
    // Tell them which global indices I need from them
    int node = -1;
    if (send_nnz > 0)
        node = dest[0] / PPN;

    first = 0;
    last = 0;
    n_sends = 0;
    for (int i = 0; i < send_nnz; i++)
    {
        proc = dest[i];
        if (proc / PPN != node)
        {
            MPI_Issend(&(node_send_buffer[first]), (last - first), MPI_BYTE,
                    node*PPN + local_rank, tag, comm->global_comm, &(comm->requests[n_sends++]));
            first = last;
            node = proc / PPN;
        }
        memcpy(&(node_send_buffer[last]), &proc, sizeof(int));
        last += sizeof(int);
        memcpy(&(node_send_buffer[last]), &(send_buffer[i*send_bytes]), send_bytes);
        last += send_bytes;
    }
    if (node >= 0)
    {
        MPI_Issend(&(node_send_buffer[first]), (last - first), MPI_BYTE,
                node*PPN + local_rank, tag, comm->global_comm, &(comm->requests[n_sends++]));
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
        MPI_Iprobe(MPI_ANY_SOURCE, tag, comm->global_comm, &flag, &recv_status);
        if (flag)
        {
            // Get the source process and message size
            proc = recv_status.MPI_SOURCE;
            MPI_Get_count(&recv_status, MPI_BYTE, &count);
            recv_buf.resize(ctr + count);

            // Receive the message, and add local indices to send_comm
            MPI_Recv(&(recv_buf[ctr]), count, MPI_BYTE, proc, tag, comm->global_comm, 
                    &recv_status);
            ctr += count;


            n_msgs = count / (recv_bytes + sizeof(int));
            for (int i = 0; i < n_msgs; i++)
                origins.push_back(proc);

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
                MPI_Ibarrier(comm->global_comm, &bar_req);
            }
        }
    }

    std::vector<int> msg_counts(PPN, 0);
    ctr = 0;
    while (ctr < recv_buf.size())
    {
        memcpy(&proc, &(recv_buf[ctr]), sizeof(int));
        proc -= (comm->rank_node * PPN);
        ctr += recv_bytes + sizeof(int);
        msg_counts[proc] += recv_bytes + sizeof(int);
    }

    std::vector<int> displs(PPN+1);
    displs[0] = 0;
    for (int i = 0; i < PPN; i++)
        displs[i+1] = displs[i] + msg_counts[i];
    
    count = recv_buf.size();
    if (count)
        local_send_buffer.resize(count);

    ctr = 0;
    idx = 0;
    while (ctr < recv_buf.size())
    {
        memcpy(&proc, &(recv_buf[ctr]), sizeof(int));
        proc -= (comm->rank_node * PPN);

        memcpy(&(local_send_buffer[displs[proc]]), &(origins[idx]), sizeof(int));
        ctr += sizeof(int);
        displs[proc] += sizeof(int);

        memcpy(&(local_send_buffer[displs[proc]]), &(recv_buf[ctr]), recv_bytes);
        ctr += recv_bytes;
        displs[proc] += recv_bytes;
        idx++;
    }
    displs[0] = 0;
    for (int i = 0; i < PPN; i++)
        displs[i+1] = displs[i] + msg_counts[i];

    tag = xinfo->tag;
    xinfo->tag = (xinfo->tag + 1 % MPI_TAG_UB);

    MPI_Allreduce(MPI_IN_PLACE, msg_counts.data(), PPN, MPI_INT, MPI_SUM, comm->local_comm);
    int recv_count = msg_counts[local_rank];
    if (PPN > comm->n_requests)
        MPIX_Comm_req_resize(comm, PPN);

    // Send a message to every process that I will need data from
    // Tell them which global indices I need from them
    n_sends = 0;
    for (int i = 0; i < PPN; i++)
    {
        if (displs[i+1] == displs[i])
            continue;

        MPI_Isend(&(local_send_buffer[displs[i]]), displs[i+1] - displs[i], MPI_BYTE, i, tag,
                comm->local_comm, &(comm->requests[n_sends++]));
    }

    count = recv_count * (recv_bytes + sizeof(int));
    if (count)
        local_recv_buffer.resize(count);

    // Wait to receive values
    // until I have received fewer than the number of global indices I am waiting on
    ctr = 0;
    while(ctr < recv_count)
    {
        // Wait for a message
        MPI_Probe(MPI_ANY_SOURCE, tag, comm->local_comm, &recv_status);

        // Get the source process and message size
        proc = recv_status.MPI_SOURCE;
        MPI_Get_count(&recv_status, MPI_BYTE, &count);

        // Receive the message, and add local indices to send_comm
        MPI_Recv(&(local_recv_buffer[ctr]), count, MPI_BYTE, proc, tag, 
                comm->local_comm, MPI_STATUS_IGNORE);
        ctr += count;
    }
    if (n_sends) MPI_Waitall(n_sends, comm->requests, MPI_STATUSES_IGNORE);

    // Last Step : Step through recvbuf to find proc of origin, size, and indices
    idx = 0;
    new_idx = 0;
    n_recvs = 0;
    while (idx < ctr)
    {
        memcpy(&proc, &(local_recv_buffer[idx]), sizeof(int));
        src[n_recvs++] = proc;
        idx += sizeof(int);
        memcpy(&(recv_buffer[new_idx]), &(local_recv_buffer[idx]), recv_bytes);
        idx += recv_bytes;
        new_idx += recv_bytes;
    }
    *recv_nnz = n_recvs;

    return MPI_SUCCESS;
}






int alltoallv_crs_personalized_loc(int send_nnz, int send_size, int* dest, int* sendcounts,
        int* sdispls, MPI_Datatype sendtype, void* sendvals,
        int* recv_nnz, int* recv_size, int* src, int* recvcounts, 
        int* rdispls, MPI_Datatype recvtype, void* recvvals, MPIX_Info* xinfo, MPIX_Comm* comm)
{
    int rank, num_procs, local_rank, PPN;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    if (comm->local_comm == MPI_COMM_NULL)
        MPIX_Comm_topo_init(comm);

    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &PPN);

    char* send_buffer = (char*)sendvals;
    char* recv_buffer = (char*)recvvals;
    int send_bytes, recv_bytes, bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    if (comm->n_requests < send_nnz)
        MPIX_Comm_req_resize(comm, send_nnz);

    int tag = xinfo->tag;
    xinfo->tag = (xinfo->tag + 1 % MPI_TAG_UB);
    MPI_Status recv_status;
    MPI_Request bar_req;
    int proc, ctr, flag, ibar;
    int first, last, count, n_msgs, size;
    int n_sends, n_recvs, idx, new_idx;
    int origin, msg_count, size_recvs;

    std::vector<char> node_send_buffer;
    std::vector<char> local_send_buffer;
    std::vector<char> local_recv_buffer;

    count = send_size*send_bytes + 2*send_nnz*sizeof(int);
    if (count)
        node_send_buffer.resize(count);

    // Send a message to every process that I will need data from
    // Tell them which global indices I need from them
    std::vector<int> msg_counts(num_procs, 0);
    int node = -1;
    for (int i = 0; i < send_nnz; i++)
    {
        proc = (dest[i] / PPN) * PPN + local_rank;
        msg_counts[proc] += 2*sizeof(int) + (sendcounts[i] * send_bytes);
    }
    MPI_Allreduce(MPI_IN_PLACE, msg_counts.data(), num_procs, MPI_INT, MPI_SUM, comm->global_comm);
    int node_recv_size = msg_counts[rank];


    first = 0;
    last = 0;
    n_sends = 0;
    if (send_nnz > 0)
        node = dest[0] / PPN;
    for (int i = 0; i < send_nnz; i++)
    {
        proc = dest[i];
        if (proc / PPN != node)
        {
            MPI_Isend(&(node_send_buffer[first]), (last - first), MPI_BYTE,
                    node*PPN + local_rank, tag, comm->global_comm, &(comm->requests[n_sends++]));
            first = last;
            node = proc / PPN;
        }
        memcpy(&(node_send_buffer[last]), &proc, sizeof(int));
        last += sizeof(int);
        size = sendcounts[i] * send_bytes;
        memcpy(&(node_send_buffer[last]), &(size), sizeof(int));
        last += sizeof(int);
        memcpy(&(node_send_buffer[last]), &(send_buffer[sdispls[i]*send_bytes]), 
                sendcounts[i]*send_bytes);
        last += sendcounts[i]*send_bytes;
    }
    if (node >= 0)
    {
        MPI_Isend(&(node_send_buffer[first]), (last - first), MPI_BYTE,
                node*PPN + local_rank, tag, comm->global_comm, &(comm->requests[n_sends++]));
    }


    std::vector<char> recv_buf;
    if (node_recv_size)
        recv_buf.resize(node_recv_size);

    std::vector<int> recv_procs;
    std::vector<int> recv_counts;
    ibar = 0;
    ctr = 0;
    // Wait to receive values
    // until I have received fewer than the number of global indices I am waiting on
    while (ctr < node_recv_size)
    {
        // Wait for a message
        MPI_Probe(MPI_ANY_SOURCE, tag, comm->global_comm, &recv_status);
            
        // Get the source process and message size
        MPI_Get_count(&recv_status, MPI_BYTE, &count);
        proc = recv_status.MPI_SOURCE;

        // Receive the message, and add local indices to send_comm
        MPI_Recv(&(recv_buf[ctr]), count, MPI_BYTE, proc, tag, comm->global_comm, 
                &recv_status);
        ctr += count;

        recv_procs.push_back(proc);
        recv_counts.push_back(count);
    }

    msg_counts.resize(PPN);
    std::fill(msg_counts.begin(), msg_counts.end(), 0);

    ctr = 0;
    while (ctr < recv_buf.size())
    {
        memcpy(&proc, &(recv_buf[ctr]), sizeof(int));
        ctr += sizeof(int);
        memcpy(&size, &(recv_buf[ctr]), sizeof(int));
        ctr += size + sizeof(int);

        proc -= (comm->rank_node * PPN);
        msg_counts[proc] += size + 2*sizeof(int);
    }

    std::vector<int> displs(PPN+1);
    displs[0] = 0;
    for (int i = 0; i < PPN; i++)
        displs[i+1] = displs[i] + msg_counts[i];
    
    count = recv_buf.size();
    if (count)
        local_send_buffer.resize(count);

    ctr = 0;
    for (idx = 0; idx < recv_procs.size(); idx++)
    {
        origin = recv_procs[idx];
        msg_count = recv_counts[idx];
        count = 0;
        while (count < msg_count)
        {
            memcpy(&proc, &(recv_buf[ctr]), sizeof(int));
            memcpy(&(recv_buf[ctr]), &origin, sizeof(int));
            memcpy(&size, &(recv_buf[ctr + sizeof(int)]), sizeof(int));

            proc -= (comm->rank_node * PPN);

            bytes = size + 2*sizeof(int);
            memcpy(&(local_send_buffer[displs[proc]]), &(recv_buf[ctr]), bytes);

            ctr += bytes;
            displs[proc] += bytes;
            count += bytes;
        }
    }

    displs[0] = 0;
    for (int i = 0; i < PPN; i++)
        displs[i+1] = displs[i] + msg_counts[i];

    tag = xinfo->tag;
    xinfo->tag = (xinfo->tag + 1 % MPI_TAG_UB);

    MPI_Allreduce(MPI_IN_PLACE, msg_counts.data(), PPN, MPI_INT, MPI_SUM, comm->local_comm);
    int recv_count = msg_counts[local_rank];
    if (PPN > comm->n_requests)
        MPIX_Comm_req_resize(comm, PPN);

    // Send a message to every process that I will need data from
    // Tell them which global indices I need from them
    n_sends = 0;
    for (int i = 0; i < PPN; i++)
    {
        if (displs[i+1] == displs[i])
            continue;

        MPI_Isend(&(local_send_buffer[displs[i]]), displs[i+1] - displs[i], MPI_BYTE, i, tag,
                comm->local_comm, &(comm->requests[n_sends++]));
    }

    if (recv_count)
        local_recv_buffer.resize(recv_count);

    // Wait to receive values
    // until I have received fewer than the number of global indices I am waiting on
    ctr = 0;
    while(ctr < recv_count)
    {
        // Wait for a message
        MPI_Probe(MPI_ANY_SOURCE, tag, comm->local_comm, &recv_status);

        // Get the source process and message size
        proc = recv_status.MPI_SOURCE;
        MPI_Get_count(&recv_status, MPI_BYTE, &count);

        // Receive the message, and add local indices to send_comm
        MPI_Recv(&(local_recv_buffer[ctr]), count, MPI_BYTE, proc, tag, 
                comm->local_comm, MPI_STATUS_IGNORE);
        ctr += count;
    }
    if (n_sends) MPI_Waitall(n_sends, comm->requests, MPI_STATUSES_IGNORE);

    // Last Step : Step through recvbuf to find proc of origin, size, and indices
    idx = 0;
    new_idx = 0;
    n_recvs = 0;
    size_recvs = 0;
    rdispls[0] = 0;
    while (idx < recv_count)
    {
        memcpy(&proc, &(local_recv_buffer[idx]), sizeof(int));
        idx += sizeof(int);
        memcpy(&size, &(local_recv_buffer[idx]), sizeof(int));
        idx += sizeof(int);
        src[n_recvs] = proc;
        recvcounts[n_recvs] = size / recv_bytes;
        rdispls[n_recvs+1] = rdispls[n_recvs] + recvcounts[n_recvs];
        size_recvs += recvcounts[n_recvs];
        n_recvs++;
        memcpy(&(recv_buffer[new_idx]), &(local_recv_buffer[idx]), size);
        idx += size;
        new_idx += size;
    }
    *recv_nnz = n_recvs;
    *recv_size = size_recvs;

    return MPI_SUCCESS;
}


int alltoallv_crs_nonblocking_loc(int send_nnz, int send_size, int* dest, int* sendcounts,
        int* sdispls, MPI_Datatype sendtype, void* sendvals,
        int* recv_nnz, int* recv_size, int* src, int* recvcounts, 
        int* rdispls, MPI_Datatype recvtype, void* recvvals, MPIX_Info* xinfo, MPIX_Comm* comm)
{
    int rank, num_procs, local_rank, PPN;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    if (comm->local_comm == MPI_COMM_NULL)
        MPIX_Comm_topo_init(comm);

    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &PPN);

    char* send_buffer = (char*)sendvals;
    char* recv_buffer = (char*)recvvals;
    int send_bytes, recv_bytes, bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    if (comm->n_requests < send_nnz)
        MPIX_Comm_req_resize(comm, send_nnz);

    int tag = xinfo->tag;
    xinfo->tag = (xinfo->tag + 1 % MPI_TAG_UB);
    MPI_Status recv_status;
    MPI_Request bar_req;
    int proc, ctr, flag, ibar;
    int first, last, count, n_msgs, size;
    int n_sends, n_recvs, idx, new_idx;
    int origin, msg_count, size_recvs;

    std::vector<char> node_send_buffer;
    std::vector<char> local_send_buffer;
    std::vector<char> local_recv_buffer;

    count = send_size*send_bytes + 2*send_nnz*sizeof(int);
    if (count)
        node_send_buffer.resize(count);

    // Send a message to every process that I will need data from
    // Tell them which global indices I need from them
    std::vector<int> msg_counts(num_procs, 0);
    int node = -1;

    first = 0;
    last = 0;
    n_sends = 0;
    if (send_nnz > 0)
        node = dest[0] / PPN;
    for (int i = 0; i < send_nnz; i++)
    {
        proc = dest[i];
        if (proc / PPN != node)
        {
            MPI_Issend(&(node_send_buffer[first]), (last - first), MPI_BYTE,
                    node*PPN + local_rank, tag, comm->global_comm, &(comm->requests[n_sends++]));
            first = last;
            node = proc / PPN;
        }
        memcpy(&(node_send_buffer[last]), &proc, sizeof(int));
        last += sizeof(int);
        size = sendcounts[i] * send_bytes;
        memcpy(&(node_send_buffer[last]), &(size), sizeof(int));
        last += sizeof(int);
        memcpy(&(node_send_buffer[last]), &(send_buffer[sdispls[i]*send_bytes]), 
                sendcounts[i]*send_bytes);
        last += sendcounts[i]*send_bytes;
    }
    if (node >= 0)
    {
        MPI_Issend(&(node_send_buffer[first]), (last - first), MPI_BYTE,
                node*PPN + local_rank, tag, comm->global_comm, &(comm->requests[n_sends++]));
    }


    std::vector<char> recv_buf;
    std::vector<int> recv_procs;
    std::vector<int> recv_counts;
    ibar = 0;
    ctr = 0;
    // Wait to receive values
    // until I have received fewer than the number of global indices I am waiting on
    while (1)
    {
        // Wait for a message
        MPI_Iprobe(MPI_ANY_SOURCE, tag, comm->global_comm, &flag, &recv_status);
            
        if (flag)
        {
            // Get the source process and message size
            MPI_Get_count(&recv_status, MPI_BYTE, &count);
            proc = recv_status.MPI_SOURCE;

            // Receive the message, and add local indices to send_comm
            recv_buf.resize(ctr + count);
            MPI_Recv(&(recv_buf[ctr]), count, MPI_BYTE, proc, tag, comm->global_comm, 
                    &recv_status);
            ctr += count;

            recv_procs.push_back(proc);
            recv_counts.push_back(count);
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
                MPI_Ibarrier(comm->global_comm, &bar_req);
            }
        }
    }

    msg_counts.resize(PPN);
    std::fill(msg_counts.begin(), msg_counts.end(), 0);

    ctr = 0;
    while (ctr < recv_buf.size())
    {
        memcpy(&proc, &(recv_buf[ctr]), sizeof(int));
        ctr += sizeof(int);
        memcpy(&size, &(recv_buf[ctr]), sizeof(int));
        ctr += size + sizeof(int);

        proc -= (comm->rank_node * PPN);
        msg_counts[proc] += size + 2*sizeof(int);
    }

    std::vector<int> displs(PPN+1);
    displs[0] = 0;
    for (int i = 0; i < PPN; i++)
        displs[i+1] = displs[i] + msg_counts[i];
    
    count = recv_buf.size();
    if (count)
        local_send_buffer.resize(count);

    ctr = 0;
    for (idx = 0; idx < recv_procs.size(); idx++)
    {
        origin = recv_procs[idx];
        msg_count = recv_counts[idx];
        count = 0;
        while (count < msg_count)
        {
            memcpy(&proc, &(recv_buf[ctr]), sizeof(int));
            memcpy(&(recv_buf[ctr]), &origin, sizeof(int));
            memcpy(&size, &(recv_buf[ctr + sizeof(int)]), sizeof(int));

            proc -= (comm->rank_node * PPN);

            bytes = size + 2*sizeof(int);
            memcpy(&(local_send_buffer[displs[proc]]), &(recv_buf[ctr]), bytes);

            ctr += bytes;
            displs[proc] += bytes;
            count += bytes;
        }
    }

    displs[0] = 0;
    for (int i = 0; i < PPN; i++)
        displs[i+1] = displs[i] + msg_counts[i];

    tag = xinfo->tag;
    xinfo->tag = (xinfo->tag + 1 % MPI_TAG_UB);

    MPI_Allreduce(MPI_IN_PLACE, msg_counts.data(), PPN, MPI_INT, MPI_SUM, comm->local_comm);
    int recv_count = msg_counts[local_rank];
    if (PPN > comm->n_requests)
        MPIX_Comm_req_resize(comm, PPN);

    // Send a message to every process that I will need data from
    // Tell them which global indices I need from them
    n_sends = 0;
    for (int i = 0; i < PPN; i++)
    {
        if (displs[i+1] == displs[i])
            continue;

        MPI_Isend(&(local_send_buffer[displs[i]]), displs[i+1] - displs[i], MPI_BYTE, i, tag,
                comm->local_comm, &(comm->requests[n_sends++]));
    }

    if (recv_count)
        local_recv_buffer.resize(recv_count);

    // Wait to receive values
    // until I have received fewer than the number of global indices I am waiting on
    ctr = 0;
    while(ctr < recv_count)
    {
        // Wait for a message
        MPI_Probe(MPI_ANY_SOURCE, tag, comm->local_comm, &recv_status);

        // Get the source process and message size
        proc = recv_status.MPI_SOURCE;
        MPI_Get_count(&recv_status, MPI_BYTE, &count);

        // Receive the message, and add local indices to send_comm
        MPI_Recv(&(local_recv_buffer[ctr]), count, MPI_BYTE, proc, tag, 
                comm->local_comm, MPI_STATUS_IGNORE);
        ctr += count;
    }
    if (n_sends) MPI_Waitall(n_sends, comm->requests, MPI_STATUSES_IGNORE);

    // Last Step : Step through recvbuf to find proc of origin, size, and indices
    idx = 0;
    new_idx = 0;
    n_recvs = 0;
    size_recvs = 0;
    rdispls[0] = 0;
    while (idx < recv_count)
    {
        memcpy(&proc, &(local_recv_buffer[idx]), sizeof(int));
        idx += sizeof(int);
        memcpy(&size, &(local_recv_buffer[idx]), sizeof(int));
        idx += sizeof(int);
        src[n_recvs] = proc;
        recvcounts[n_recvs] = size / recv_bytes;
        rdispls[n_recvs+1] = rdispls[n_recvs] + recvcounts[n_recvs];
        size_recvs += recvcounts[n_recvs];
        n_recvs++;
        memcpy(&(recv_buffer[new_idx]), &(local_recv_buffer[idx]), size);
        idx += size;
        new_idx += size;
    }
    *recv_nnz = n_recvs;
    *recv_size = size_recvs;

    return MPI_SUCCESS;
}

