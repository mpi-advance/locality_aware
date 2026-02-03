#include <cstring>
#include <vector>

#include "communicator/MPIL_Comm.h"
#include "locality_aware.h"
// Assumes SMP Ordering of ranks across nodes (aggregates ranks 0-PPN)
int alltoall_crs_personalized_loc(int send_nnz,
                                  int* dest,
                                  int sendcount,
                                  MPI_Datatype sendtype,
                                  void* sendvals,
                                  int* recv_nnz,
                                  int* src,
                                  int recvcount,
                                  MPI_Datatype recvtype,
                                  void* recvvals,
                                  MPIL_Info* xinfo,
                                  MPIL_Comm* comm)
{
    if (comm->local_comm == MPI_COMM_NULL)
    {
        MPIL_Comm_topo_init(comm);
    }

    int local_rank, PPN;
    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &PPN);

    char* send_buffer = (char*)sendvals;
    char* recv_buffer = (char*)recvvals;
    int send_bytes, recv_bytes, int_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);
    MPI_Type_size(MPI_INT, &int_bytes);
    send_bytes *= sendcount;
    recv_bytes *= recvcount;

    if (comm->n_requests < send_nnz)
    {
        MPIL_Comm_req_resize(comm, send_nnz);
    }

    MPI_Status recv_status;
    int proc, ctr, start, end;
    int count, n_msgs, n_sends, n_recvs, idx, new_idx;
    int tag;
    get_tag(comm, &tag);

    std::vector<char> node_send_buffer;
    std::vector<char> local_send_buffer;
    std::vector<char> local_recv_buffer;

    count = send_nnz * (send_bytes + int_bytes);
    if (count)
    {
        node_send_buffer.resize(count);
    }

    // Send a message to every process that I will need data from
    // Tell them which global indices I need from them
    int group_procs, group_rank;
    ;
    MPI_Comm_size(comm->group_comm, &group_procs);
    MPI_Comm_rank(comm->group_comm, &group_rank);

    std::vector<int> msg_counts(group_procs, 0);
    std::vector<int> msg_displs(group_procs + 1);
    int node = -1;
    for (int i = 0; i < send_nnz; i++)
    {
        proc = dest[i] / PPN;
        msg_counts[proc] += int_bytes + send_bytes;
    }
    msg_displs[0] = 0;
    for (int i = 0; i < group_procs; i++)
    {
        msg_displs[i + 1] = msg_displs[i] + msg_counts[i];
    }

    for (int i = 0; i < send_nnz; i++)
    {
        proc = dest[i];
        if (proc / PPN != node)
        {
            node = proc / PPN;
        }
        MPI_Pack(&proc,
                 1,
                 MPI_INT,
                 node_send_buffer.data(),
                 node_send_buffer.size(),
                 &(msg_displs[node]),
                 comm->group_comm);
        MPI_Pack(&(send_buffer[i * send_bytes]),
                 sendcount,
                 sendtype,
                 node_send_buffer.data(),
                 node_send_buffer.size(),
                 &(msg_displs[node]),
                 comm->group_comm);
    }
    msg_displs[0] = 0;
    for (int i = 0; i < group_procs; i++)
    {
        msg_displs[i + 1] = msg_displs[i] + msg_counts[i];
    }

    MPI_Allreduce(
        MPI_IN_PLACE, msg_counts.data(), group_procs, MPI_INT, MPI_SUM, comm->group_comm);
    int node_recv_size = msg_counts[group_rank];

    if (send_nnz > 0)
    {
        node = dest[0] / PPN;
    }

    n_sends = 0;
    for (int i = 0; i < group_procs; i++)
    {
        start = msg_displs[i];
        end   = msg_displs[i + 1];
        if (end - start)
        {
            MPI_Isend(&(node_send_buffer[start]),
                      end - start,
                      MPI_PACKED,
                      i,
                      tag,
                      comm->group_comm,
                      &(comm->requests[n_sends++]));
        }
    }

    std::vector<char> recv_buf;
    if (node_recv_size)
    {
        recv_buf.resize(node_recv_size);
    }

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
        // recv_buf.resize(ctr + count);

        // Receive the message, and add local indices to send_comm
        MPI_Recv(&(recv_buf[ctr]),
                 count,
                 MPI_PACKED,
                 proc,
                 tag,
                 comm->group_comm,
                 &recv_status);
        ctr += count;

        n_msgs = count / (recv_bytes + int_bytes);
        for (int i = 0; i < n_msgs; i++)
        {
            origins.push_back(proc * PPN + local_rank);
        }
    }

    msg_counts.resize(PPN);
    std::fill(msg_counts.begin(), msg_counts.end(), 0);

    ctr = 0;
    while (ctr < node_recv_size)
    {
        MPI_Unpack(
            recv_buf.data(), node_recv_size, &ctr, &proc, 1, MPI_INT, comm->group_comm);
        proc -= (comm->rank_node * PPN);
        ctr += recv_bytes;
        msg_counts[proc] += recv_bytes + int_bytes;
    }

    std::vector<int> displs(PPN + 1);
    displs[0] = 0;
    for (int i = 0; i < PPN; i++)
    {
        displs[i + 1] = displs[i] + msg_counts[i];
    }

    count = recv_buf.size();
    if (count)
    {
        local_send_buffer.resize(count);
    }

    ctr = 0;
    idx = 0;
    while (ctr < node_recv_size)
    {
        MPI_Unpack(
            recv_buf.data(), node_recv_size, &ctr, &proc, 1, MPI_INT, comm->group_comm);
        proc -= (comm->rank_node * PPN);

        MPI_Pack(&(origins[idx++]),
                 1,
                 MPI_INT,
                 local_send_buffer.data(),
                 local_send_buffer.size(),
                 &(displs[proc]),
                 comm->local_comm);
        MPI_Pack(&(recv_buf[ctr]),
                 recv_bytes,
                 MPI_PACKED,
                 local_send_buffer.data(),
                 local_send_buffer.size(),
                 &(displs[proc]),
                 comm->local_comm);
        ctr += recv_bytes;
    }
    displs[0] = 0;
    for (int i = 0; i < PPN; i++)
    {
        displs[i + 1] = displs[i] + msg_counts[i];
    }

    get_tag(comm, &tag);

    MPI_Allreduce(
        MPI_IN_PLACE, msg_counts.data(), PPN, MPI_INT, MPI_SUM, comm->local_comm);
    int recv_count = msg_counts[local_rank];
    if (PPN > comm->n_requests)
    {
        MPIL_Comm_req_resize(comm, PPN);
    }

    // Send a message to every process that I will need data from
    // Tell them which global indices I need from them
    n_sends = 0;
    for (int i = 0; i < PPN; i++)
    {
        if (displs[i + 1] == displs[i])
        {
            continue;
        }

        MPI_Isend(&(local_send_buffer[displs[i]]),
                  displs[i + 1] - displs[i],
                  MPI_PACKED,
                  i,
                  tag,
                  comm->local_comm,
                  &(comm->requests[n_sends++]));
    }

    count = recv_count * (recv_bytes + int_bytes);
    if (count)
    {
        local_recv_buffer.resize(count);
    }

    // Wait to receive values
    // until I have received fewer than the number of global indices I am waiting on
    ctr = 0;
    while (ctr < recv_count)
    {
        // Wait for a message
        MPI_Probe(MPI_ANY_SOURCE, tag, comm->local_comm, &recv_status);

        // Get the source process and message size
        proc = recv_status.MPI_SOURCE;
        MPI_Get_count(&recv_status, MPI_PACKED, &count);

        // Receive the message, and add local indices to send_comm
        MPI_Recv(&(local_recv_buffer[ctr]),
                 count,
                 MPI_PACKED,
                 proc,
                 tag,
                 comm->local_comm,
                 MPI_STATUS_IGNORE);
        ctr += count;
    }
    if (n_sends)
    {
        MPI_Waitall(n_sends, comm->requests, MPI_STATUSES_IGNORE);
    }

    // Last Step : Step through recvbuf to find proc of origin, size, and indices
    idx     = 0;
    new_idx = 0;
    n_recvs = 0;
    while (idx < ctr)
    {
        MPI_Unpack(local_recv_buffer.data(),
                   local_recv_buffer.size(),
                   &idx,
                   &proc,
                   1,
                   MPI_INT,
                   comm->local_comm);
        MPI_Unpack(local_recv_buffer.data(),
                   local_recv_buffer.size(),
                   &idx,
                   &(recv_buffer[new_idx]),
                   recv_bytes,
                   MPI_BYTE,
                   comm->local_comm);
        src[n_recvs++] = proc;
        new_idx += recv_bytes;
    }
    *recv_nnz = n_recvs;

    return MPI_SUCCESS;
}

// Assumes SMP Ordering of ranks across nodes (aggregates ranks 0-PPN)
int alltoall_crs_nonblocking_loc(int send_nnz,
                                 int* dest,
                                 int sendcount,
                                 MPI_Datatype sendtype,
                                 void* sendvals,
                                 int* recv_nnz,
                                 int* src,
                                 int recvcount,
                                 MPI_Datatype recvtype,
                                 void* recvvals,
                                 MPIL_Info* xinfo,
                                 MPIL_Comm* comm)
{
    if (comm->local_comm == MPI_COMM_NULL)
    {
        MPIL_Comm_topo_init(comm);
    }

    int local_rank, PPN;
    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &PPN);

    char* send_buffer = (char*)sendvals;
    char* recv_buffer = (char*)recvvals;
    int send_bytes, recv_bytes, int_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);
    MPI_Type_size(MPI_INT, &int_bytes);
    send_bytes *= sendcount;
    recv_bytes *= recvcount;

    if (comm->n_requests < send_nnz)
    {
        MPIL_Comm_req_resize(comm, send_nnz);
    }

    MPI_Status recv_status;
    MPI_Request bar_req;
    int proc, ctr, flag, ibar, start, end;
    int count, n_msgs, n_sends, n_recvs, idx, new_idx;
    int tag;
    get_tag(comm, &tag);

    std::vector<char> node_send_buffer;
    std::vector<char> local_send_buffer;
    std::vector<char> local_recv_buffer;

    count = send_nnz * (send_bytes + int_bytes);
    if (count)
    {
        node_send_buffer.resize(count);
    }

    int group_procs, group_rank;
    ;
    MPI_Comm_size(comm->group_comm, &group_procs);
    MPI_Comm_rank(comm->group_comm, &group_rank);

    // Send a message to every process that I will need data from
    // Tell them which global indices I need from them
    std::vector<int> msg_counts(group_procs, 0);
    std::vector<int> msg_displs(group_procs + 1);
    int node = -1;
    for (int i = 0; i < send_nnz; i++)
    {
        proc = dest[i] / PPN;
        msg_counts[proc] += int_bytes + send_bytes;
    }
    msg_displs[0] = 0;
    for (int i = 0; i < group_procs; i++)
    {
        msg_displs[i + 1] = msg_displs[i] + msg_counts[i];
    }

    for (int i = 0; i < send_nnz; i++)
    {
        proc = dest[i];
        if (proc / PPN != node)
        {
            node = proc / PPN;
        }
        MPI_Pack(&proc,
                 1,
                 MPI_INT,
                 node_send_buffer.data(),
                 node_send_buffer.size(),
                 &(msg_displs[node]),
                 comm->group_comm);
        MPI_Pack(&(send_buffer[i * send_bytes]),
                 sendcount,
                 sendtype,
                 node_send_buffer.data(),
                 node_send_buffer.size(),
                 &(msg_displs[node]),
                 comm->group_comm);
    }
    msg_displs[0] = 0;
    for (int i = 0; i < group_procs; i++)
    {
        msg_displs[i + 1] = msg_displs[i] + msg_counts[i];
    }

    if (send_nnz > 0)
    {
        node = dest[0] / PPN;
    }

    n_sends = 0;
    for (int i = 0; i < group_procs; i++)
    {
        start = msg_displs[i];
        end   = msg_displs[i + 1];
        if (end - start)
        {
            MPI_Issend(&(node_send_buffer[start]),
                       end - start,
                       MPI_PACKED,
                       i,
                       tag,
                       comm->group_comm,
                       &(comm->requests[n_sends++]));
        }
    }

    std::vector<char> recv_buf;
    std::vector<int> origins;
    ibar = 0;
    ctr  = 0;
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
            MPI_Recv(&(recv_buf[ctr]),
                     count,
                     MPI_PACKED,
                     proc,
                     tag,
                     comm->group_comm,
                     &recv_status);
            ctr += count;

            n_msgs = count / (recv_bytes + int_bytes);
            for (int i = 0; i < n_msgs; i++)
            {
                origins.push_back(proc * PPN + local_rank);
            }
        }

        // If I have already called my Ibarrier, check if all processes have reached
        // If all processes have reached the Ibarrier, all messages have been sent
        if (ibar)
        {
            MPI_Test(&bar_req, &flag, &recv_status);
            if (flag)
            {
                break;
            }
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
    msg_counts.resize(PPN);
    std::fill(msg_counts.begin(), msg_counts.end(), 0);
    ctr = 0;
    while (ctr < node_recv_size)
    {
        MPI_Unpack(
            recv_buf.data(), node_recv_size, &ctr, &proc, 1, MPI_INT, comm->group_comm);
        proc -= (comm->rank_node * PPN);
        ctr += recv_bytes;
        msg_counts[proc] += recv_bytes + int_bytes;
    }

    std::vector<int> displs(PPN + 1);
    displs[0] = 0;
    for (int i = 0; i < PPN; i++)
    {
        displs[i + 1] = displs[i] + msg_counts[i];
    }

    count = recv_buf.size();
    if (count)
    {
        local_send_buffer.resize(count);
    }

    ctr = 0;
    idx = 0;
    while (ctr < node_recv_size)
    {
        MPI_Unpack(
            recv_buf.data(), node_recv_size, &ctr, &proc, 1, MPI_INT, comm->group_comm);
        proc -= (comm->rank_node * PPN);

        MPI_Pack(&(origins[idx++]),
                 1,
                 MPI_INT,
                 local_send_buffer.data(),
                 local_send_buffer.size(),
                 &(displs[proc]),
                 comm->local_comm);
        MPI_Pack(&(recv_buf[ctr]),
                 recv_bytes,
                 MPI_PACKED,
                 local_send_buffer.data(),
                 local_send_buffer.size(),
                 &(displs[proc]),
                 comm->local_comm);
        ctr += recv_bytes;
    }
    displs[0] = 0;
    for (int i = 0; i < PPN; i++)
    {
        displs[i + 1] = displs[i] + msg_counts[i];
    }

    get_tag(comm, &tag);

    MPI_Allreduce(
        MPI_IN_PLACE, msg_counts.data(), PPN, MPI_INT, MPI_SUM, comm->local_comm);
    int recv_count = msg_counts[local_rank];
    if (PPN > comm->n_requests)
    {
        MPIL_Comm_req_resize(comm, PPN);
    }

    // Send a message to every process that I will need data from
    // Tell them which global indices I need from them
    n_sends = 0;
    for (int i = 0; i < PPN; i++)
    {
        if (displs[i + 1] == displs[i])
        {
            continue;
        }

        MPI_Isend(&(local_send_buffer[displs[i]]),
                  displs[i + 1] - displs[i],
                  MPI_PACKED,
                  i,
                  tag,
                  comm->local_comm,
                  &(comm->requests[n_sends++]));
    }

    count = recv_count * (recv_bytes + int_bytes);
    if (count)
    {
        local_recv_buffer.resize(count);
    }

    // Wait to receive values
    // until I have received fewer than the number of global indices I am waiting on
    ctr = 0;
    while (ctr < recv_count)
    {
        // Wait for a message
        MPI_Probe(MPI_ANY_SOURCE, tag, comm->local_comm, &recv_status);

        // Get the source process and message size
        proc = recv_status.MPI_SOURCE;
        MPI_Get_count(&recv_status, MPI_PACKED, &count);

        // Receive the message, and add local indices to send_comm
        MPI_Recv(&(local_recv_buffer[ctr]),
                 count,
                 MPI_PACKED,
                 proc,
                 tag,
                 comm->local_comm,
                 MPI_STATUS_IGNORE);
        ctr += count;
    }
    if (n_sends)
    {
        MPI_Waitall(n_sends, comm->requests, MPI_STATUSES_IGNORE);
    }

    // Last Step : Step through recvbuf to find proc of origin, size, and indices
    idx     = 0;
    new_idx = 0;
    n_recvs = 0;
    while (idx < ctr)
    {
        MPI_Unpack(local_recv_buffer.data(),
                   local_recv_buffer.size(),
                   &idx,
                   &proc,
                   1,
                   MPI_INT,
                   comm->local_comm);
        MPI_Unpack(local_recv_buffer.data(),
                   local_recv_buffer.size(),
                   &idx,
                   &(recv_buffer[new_idx]),
                   recv_bytes,
                   MPI_PACKED,
                   comm->local_comm);
        src[n_recvs++] = proc;
        new_idx += recv_bytes;
    }
    *recv_nnz = n_recvs;

    return MPI_SUCCESS;
}

int alltoallv_crs_personalized_loc(int send_nnz,
                                   int send_size,
                                   int* dest,
                                   int* sendcounts,
                                   int* sdispls,
                                   MPI_Datatype sendtype,
                                   void* sendvals,
                                   int* recv_nnz,
                                   int* recv_size,
                                   int* src,
                                   int* recvcounts,
                                   int* rdispls,
                                   MPI_Datatype recvtype,
                                   void* recvvals,
                                   MPIL_Info* xinfo,
                                   MPIL_Comm* comm)
{
    if (comm->local_comm == MPI_COMM_NULL)
    {
        MPIL_Comm_topo_init(comm);
    }

    int local_rank, PPN;
    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &PPN);

    if (comm->n_requests < send_nnz)
    {
        MPIL_Comm_req_resize(comm, send_nnz);
    }

    int tag;
    get_tag(comm, &tag);

    char* send_buffer = (char*)sendvals;
    char* recv_buffer = (char*)recvvals;
    int send_bytes, recv_bytes, int_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);
    MPI_Type_size(MPI_INT, &int_bytes);

    std::vector<char> node_send_buffer(send_size * send_bytes + 2 * send_nnz * int_bytes);
    std::vector<int> sizes(PPN, 0);
    int proc, count, ctr, start, end;
    MPI_Status recv_status;

    int group_procs, group_rank;
    ;
    MPI_Comm_size(comm->group_comm, &group_procs);
    MPI_Comm_rank(comm->group_comm, &group_rank);

    // Send a message to every process that I will need data from
    // Tell them which global indices I need from them
    std::vector<int> msg_counts(group_procs, 0);
    std::vector<int> msg_displs(group_procs + 1);
    int node = -1;
    for (int i = 0; i < send_nnz; i++)
    {
        proc = dest[i] / PPN;
        msg_counts[proc] += 2 * int_bytes + sendcounts[i] * send_bytes;
    }
    msg_displs[0] = 0;
    for (int i = 0; i < group_procs; i++)
    {
        msg_displs[i + 1] = msg_displs[i] + msg_counts[i];
    }

    for (int i = 0; i < send_nnz; i++)
    {
        proc  = dest[i];
        int s = sendcounts[i] * send_bytes;
        if (proc / PPN != node)
        {
            node = proc / PPN;
        }
        MPI_Pack(&proc,
                 1,
                 MPI_INT,
                 node_send_buffer.data(),
                 node_send_buffer.size(),
                 &(msg_displs[node]),
                 comm->group_comm);
        MPI_Pack(&(s),
                 1,
                 MPI_INT,
                 node_send_buffer.data(),
                 node_send_buffer.size(),
                 &(msg_displs[node]),
                 comm->group_comm);
        MPI_Pack(&(send_buffer[sdispls[i] * send_bytes]),
                 sendcounts[i],
                 sendtype,
                 node_send_buffer.data(),
                 node_send_buffer.size(),
                 &(msg_displs[node]),
                 comm->group_comm);
    }
    msg_displs[0] = 0;
    for (int i = 0; i < group_procs; i++)
    {
        msg_displs[i + 1] = msg_displs[i] + msg_counts[i];
    }

    MPI_Allreduce(
        MPI_IN_PLACE, msg_counts.data(), group_procs, MPI_INT, MPI_SUM, comm->group_comm);
    int node_recv_size = msg_counts[group_rank];

    if (send_nnz > 0)
    {
        node = dest[0] / PPN;
    }

    int n_sends = 0;
    for (int i = 0; i < group_procs; i++)
    {
        start = msg_displs[i];
        end   = msg_displs[i + 1];
        if (end - start)
        {
            MPI_Isend(&(node_send_buffer[start]),
                      end - start,
                      MPI_PACKED,
                      i,
                      tag,
                      comm->group_comm,
                      &(comm->requests[n_sends++]));
        }
    }

    // Wait to receive values
    // until I have received fewer than the number of global indices I am waiting on
    //

    std::vector<int> origins;
    std::vector<int> origin_displs;
    origin_displs.push_back(0);
    int n_recvs = 0;

    std::vector<char> recv_buf;

    ctr = 0;
    while (ctr < node_recv_size)
    {
        // Wait for a message
        MPI_Probe(MPI_ANY_SOURCE, tag, comm->group_comm, &recv_status);
        // Get the source process and message size
        proc = recv_status.MPI_SOURCE;
        MPI_Get_count(&recv_status, MPI_PACKED, &count);

        recv_buf.resize((count + ctr));

        // Receive the message, and add local indices to send_comm
        MPI_Recv(&(recv_buf[ctr]),
                 count,
                 MPI_PACKED,
                 proc,
                 tag,
                 comm->group_comm,
                 MPI_STATUS_IGNORE);

        ctr += count;
        origins.push_back(proc * PPN + local_rank);
        origin_displs.push_back(ctr);

        n_recvs++;
    }

    MPI_Waitall(n_sends, comm->requests, MPI_STATUSES_IGNORE);

    std::vector<int> recv_sizes(PPN, 0);
    std::vector<int> recv_displs(PPN + 1);

    std::vector<char> local_buf;
    if (origin_displs[n_recvs])
    {
        local_buf.resize(origin_displs[n_recvs]);
    }

    int idx = 0;
    std::vector<char> recv_byte_buf;
    if (recv_buf.size())
    {
        recv_byte_buf.resize(recv_buf.size());
        MPI_Pack(recv_buf.data(),
                 recv_buf.size(),
                 MPI_BYTE,
                 recv_byte_buf.data(),
                 recv_byte_buf.size(),
                 &idx,
                 comm->local_comm);
    }

    idx = 0;
    while (idx < origin_displs[n_recvs])
    {
        MPI_Unpack(recv_byte_buf.data(),
                   recv_byte_buf.size(),
                   &idx,
                   &proc,
                   1,
                   MPI_INT,
                   comm->local_comm);
        MPI_Unpack(recv_byte_buf.data(),
                   recv_byte_buf.size(),
                   &idx,
                   &count,
                   1,
                   MPI_INT,
                   comm->local_comm);
        proc -= (comm->rank_node * PPN);
        idx += count;

        recv_sizes[proc] += count + 2 * int_bytes;
    }

    recv_displs[0] = 0;
    for (int i = 0; i < PPN; i++)
    {
        recv_displs[i + 1] = recv_displs[i] + recv_sizes[i];
    }

    idx = 0;
    for (int i = 0; i < n_recvs; i++)
    {
        int origin = origins[i];
        while (idx < origin_displs[i + 1])
        {
            MPI_Unpack(recv_byte_buf.data(),
                       recv_byte_buf.size(),
                       &idx,
                       &proc,
                       1,
                       MPI_INT,
                       comm->local_comm);
            MPI_Unpack(recv_byte_buf.data(),
                       recv_byte_buf.size(),
                       &idx,
                       &count,
                       1,
                       MPI_INT,
                       comm->local_comm);
            proc -= (comm->rank_node * PPN);

            MPI_Pack(&origin,
                     1,
                     MPI_INT,
                     local_buf.data(),
                     local_buf.size(),
                     &(recv_displs[proc]),
                     comm->local_comm);
            MPI_Pack(&count,
                     1,
                     MPI_INT,
                     local_buf.data(),
                     local_buf.size(),
                     &(recv_displs[proc]),
                     comm->local_comm);

            MPI_Pack(&(recv_byte_buf[idx]),
                     count,
                     MPI_PACKED,
                     local_buf.data(),
                     local_buf.size(),
                     &(recv_displs[proc]),
                     comm->local_comm);
            idx += count;
        }
    }

    recv_displs[0] = 0;
    for (int i = 0; i < PPN; i++)
    {
        recv_displs[i + 1] = recv_displs[i] + recv_sizes[i];
    }

    // STEP 2 : Local Communication
    MPI_Allreduce(
        MPI_IN_PLACE, recv_sizes.data(), PPN, MPI_INT, MPI_SUM, comm->local_comm);
    int local_size_msgs = recv_sizes[local_rank];

    // Send a message to every process that I will need data from
    // Tell them which global indices I need from them
    std::vector<MPI_Request> local_req(PPN);

    get_tag(comm, &tag);

    n_sends = 0;
    for (int i = 0; i < PPN; i++)
    {
        int s = recv_displs[i + 1] - recv_displs[i];
        if (s)
        {
            MPI_Isend(&(local_buf[recv_displs[i]]),
                      s,
                      MPI_PACKED,
                      i,
                      tag,
                      comm->local_comm,
                      &(local_req[n_sends++]));
        }
    }

    // Wait to receive values
    // until I have received fewer than the number of global indices I am waiting on
    std::vector<char> local_recv_buffer(local_size_msgs);

    ctr = 0;
    while (ctr < local_size_msgs)
    {
        // Wait for a message
        MPI_Probe(MPI_ANY_SOURCE, tag, comm->local_comm, &recv_status);

        // Get the source process and message size
        proc = recv_status.MPI_SOURCE;
        MPI_Get_count(&recv_status, MPI_PACKED, &count);

        // Receive the message, and add local indices to send_comm
        MPI_Recv(&(local_recv_buffer[ctr]),
                 count,
                 MPI_PACKED,
                 proc,
                 tag,
                 comm->local_comm,
                 &recv_status);
        ctr += count;
    }
    if (n_sends)
    {
        MPI_Waitall(n_sends, local_req.data(), MPI_STATUSES_IGNORE);
    }

    // Last Step : Step through recvbuf to find proc of origin, size, and indices
    rdispls[0]   = 0;
    n_recvs      = 0;
    int byte_ctr = 0;
    while (byte_ctr < local_size_msgs)
    {
        MPI_Unpack(local_recv_buffer.data(),
                   local_recv_buffer.size(),
                   &byte_ctr,
                   &(src[n_recvs]),
                   1,
                   MPI_INT,
                   comm->local_comm);
        MPI_Unpack(local_recv_buffer.data(),
                   local_recv_buffer.size(),
                   &byte_ctr,
                   &count,
                   1,
                   MPI_INT,
                   comm->local_comm);
        count = count / recv_bytes;

        MPI_Unpack(local_recv_buffer.data(),
                   local_recv_buffer.size(),
                   &byte_ctr,
                   &(recv_buffer[rdispls[n_recvs] * recv_bytes]),
                   count,
                   recvtype,
                   comm->local_comm);

        recvcounts[n_recvs]  = count;
        rdispls[n_recvs + 1] = rdispls[n_recvs] + count;
        n_recvs++;
    }
    // Set send sizes
    *recv_nnz  = n_recvs;
    *recv_size = rdispls[n_recvs];

    return MPI_SUCCESS;
}

int alltoallv_crs_nonblocking_loc(int send_nnz,
                                  int send_size,
                                  int* dest,
                                  int* sendcounts,
                                  int* sdispls,
                                  MPI_Datatype sendtype,
                                  void* sendvals,
                                  int* recv_nnz,
                                  int* recv_size,
                                  int* src,
                                  int* recvcounts,
                                  int* rdispls,
                                  MPI_Datatype recvtype,
                                  void* recvvals,
                                  MPIL_Info* xinfo,
                                  MPIL_Comm* comm)
{
    if (comm->local_comm == MPI_COMM_NULL)
    {
        MPIL_Comm_topo_init(comm);
    }

    int local_rank, PPN;
    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &PPN);

    if (comm->n_requests < send_nnz)
    {
        MPIL_Comm_req_resize(comm, send_nnz);
    }

    int tag;
    get_tag(comm, &tag);

    char* send_buffer = (char*)sendvals;
    char* recv_buffer = (char*)recvvals;
    int send_bytes, recv_bytes, int_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);
    MPI_Type_size(MPI_INT, &int_bytes);

    std::vector<char> node_send_buffer(send_size * send_bytes + 2 * send_nnz * int_bytes);
    std::vector<int> sizes(PPN, 0);
    int proc, count, ctr, flag, start, end;
    int ibar = 0;
    MPI_Status recv_status;
    MPI_Request bar_req;

    int group_procs, group_rank;
    ;
    MPI_Comm_size(comm->group_comm, &group_procs);
    MPI_Comm_rank(comm->group_comm, &group_rank);

    // Send a message to every process that I will need data from
    // Tell them which global indices I need from them
    std::vector<int> msg_counts(group_procs, 0);
    std::vector<int> msg_displs(group_procs + 1);
    int node = -1;
    for (int i = 0; i < send_nnz; i++)
    {
        proc = dest[i] / PPN;
        msg_counts[proc] += 2 * int_bytes + sendcounts[i] * send_bytes;
    }
    msg_displs[0] = 0;
    for (int i = 0; i < group_procs; i++)
    {
        msg_displs[i + 1] = msg_displs[i] + msg_counts[i];
    }

    for (int i = 0; i < send_nnz; i++)
    {
        proc  = dest[i];
        int s = sendcounts[i] * send_bytes;
        if (proc / PPN != node)
        {
            node = proc / PPN;
        }
        MPI_Pack(&proc,
                 1,
                 MPI_INT,
                 node_send_buffer.data(),
                 node_send_buffer.size(),
                 &(msg_displs[node]),
                 comm->group_comm);
        MPI_Pack(&(s),
                 1,
                 MPI_INT,
                 node_send_buffer.data(),
                 node_send_buffer.size(),
                 &(msg_displs[node]),
                 comm->group_comm);
        MPI_Pack(&(send_buffer[sdispls[i] * send_bytes]),
                 sendcounts[i],
                 sendtype,
                 node_send_buffer.data(),
                 node_send_buffer.size(),
                 &(msg_displs[node]),
                 comm->group_comm);
    }
    msg_displs[0] = 0;
    for (int i = 0; i < group_procs; i++)
    {
        msg_displs[i + 1] = msg_displs[i] + msg_counts[i];
    }

    if (send_nnz > 0)
    {
        node = dest[0] / PPN;
    }

    int n_sends = 0;
    for (int i = 0; i < group_procs; i++)
    {
        start = msg_displs[i];
        end   = msg_displs[i + 1];
        if (end - start)
        {
            MPI_Issend(&(node_send_buffer[start]),
                       end - start,
                       MPI_PACKED,
                       i,
                       tag,
                       comm->group_comm,
                       &(comm->requests[n_sends++]));
        }
    }

    // Wait to receive values
    // until I have received fewer than the number of global indices I am waiting on
    //

    std::vector<int> origins;
    std::vector<int> origin_displs;
    origin_displs.push_back(0);
    int n_recvs = 0;

    std::vector<char> recv_buf;

    ctr = 0;
    while (1)
    {
        // Wait for a message
        MPI_Iprobe(MPI_ANY_SOURCE, tag, comm->group_comm, &flag, &recv_status);
        if (flag)
        {
            // Get the source process and message size
            proc = recv_status.MPI_SOURCE;
            MPI_Get_count(&recv_status, MPI_PACKED, &count);

            recv_buf.resize((count + ctr));

            // Receive the message, and add local indices to send_comm
            MPI_Recv(&(recv_buf[ctr]),
                     count,
                     MPI_PACKED,
                     proc,
                     tag,
                     comm->group_comm,
                     MPI_STATUS_IGNORE);

            ctr += count;
            origins.push_back(proc * PPN + local_rank);
            origin_displs.push_back(ctr);

            n_recvs++;
        }

        // If I have already called my Ibarrier, check if all processes have reached
        // If all processes have reached the Ibarrier, all messages have been sent
        if (ibar)
        {
            MPI_Test(&bar_req, &flag, MPI_STATUS_IGNORE);
            if (flag)
            {
                break;
            }
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

    std::vector<int> recv_sizes(PPN, 0);
    std::vector<int> recv_displs(PPN + 1);

    std::vector<char> local_buf;
    if (origin_displs[n_recvs])
    {
        local_buf.resize(origin_displs[n_recvs]);
    }

    int idx = 0;
    std::vector<char> recv_byte_buf;
    if (recv_buf.size())
    {
        recv_byte_buf.resize(recv_buf.size());
        MPI_Pack(recv_buf.data(),
                 recv_buf.size(),
                 MPI_BYTE,
                 recv_byte_buf.data(),
                 recv_byte_buf.size(),
                 &idx,
                 comm->local_comm);
    }

    idx = 0;
    while (idx < origin_displs[n_recvs])
    {
        MPI_Unpack(recv_byte_buf.data(),
                   recv_byte_buf.size(),
                   &idx,
                   &proc,
                   1,
                   MPI_INT,
                   comm->local_comm);
        MPI_Unpack(recv_byte_buf.data(),
                   recv_byte_buf.size(),
                   &idx,
                   &count,
                   1,
                   MPI_INT,
                   comm->local_comm);
        proc -= (comm->rank_node * PPN);
        idx += count;

        recv_sizes[proc] += count + 2 * int_bytes;
    }

    recv_displs[0] = 0;
    for (int i = 0; i < PPN; i++)
    {
        recv_displs[i + 1] = recv_displs[i] + recv_sizes[i];
    }

    idx = 0;
    for (int i = 0; i < n_recvs; i++)
    {
        int origin = origins[i];
        while (idx < origin_displs[i + 1])
        {
            MPI_Unpack(recv_byte_buf.data(),
                       recv_byte_buf.size(),
                       &idx,
                       &proc,
                       1,
                       MPI_INT,
                       comm->local_comm);
            MPI_Unpack(recv_byte_buf.data(),
                       recv_byte_buf.size(),
                       &idx,
                       &count,
                       1,
                       MPI_INT,
                       comm->local_comm);
            proc -= (comm->rank_node * PPN);

            MPI_Pack(&origin,
                     1,
                     MPI_INT,
                     local_buf.data(),
                     local_buf.size(),
                     &(recv_displs[proc]),
                     comm->local_comm);
            MPI_Pack(&count,
                     1,
                     MPI_INT,
                     local_buf.data(),
                     local_buf.size(),
                     &(recv_displs[proc]),
                     comm->local_comm);

            MPI_Pack(&(recv_byte_buf[idx]),
                     count,
                     MPI_PACKED,
                     local_buf.data(),
                     local_buf.size(),
                     &(recv_displs[proc]),
                     comm->local_comm);
            idx += count;
        }
    }

    recv_displs[0] = 0;
    for (int i = 0; i < PPN; i++)
    {
        recv_displs[i + 1] = recv_displs[i] + recv_sizes[i];
    }

    // STEP 2 : Local Communication
    MPI_Allreduce(
        MPI_IN_PLACE, recv_sizes.data(), PPN, MPI_INT, MPI_SUM, comm->local_comm);
    int local_size_msgs = recv_sizes[local_rank];

    // Send a message to every process that I will need data from
    // Tell them which global indices I need from them
    std::vector<MPI_Request> local_req(PPN);

    get_tag(comm, &tag);

    n_sends = 0;
    for (int i = 0; i < PPN; i++)
    {
        int s = recv_displs[i + 1] - recv_displs[i];
        if (s)
        {
            MPI_Isend(&(local_buf[recv_displs[i]]),
                      s,
                      MPI_PACKED,
                      i,
                      tag,
                      comm->local_comm,
                      &(local_req[n_sends++]));
        }
    }

    // Wait to receive values
    // until I have received fewer than the number of global indices I am waiting on
    std::vector<char> local_recv_buffer(local_size_msgs);

    ctr = 0;
    while (ctr < local_size_msgs)
    {
        // Wait for a message
        MPI_Probe(MPI_ANY_SOURCE, tag, comm->local_comm, &recv_status);

        // Get the source process and message size
        proc = recv_status.MPI_SOURCE;
        MPI_Get_count(&recv_status, MPI_PACKED, &count);

        // Receive the message, and add local indices to send_comm
        MPI_Recv(&(local_recv_buffer[ctr]),
                 count,
                 MPI_PACKED,
                 proc,
                 tag,
                 comm->local_comm,
                 &recv_status);
        ctr += count;
    }
    if (n_sends)
    {
        MPI_Waitall(n_sends, local_req.data(), MPI_STATUSES_IGNORE);
    }

    // Last Step : Step through recvbuf to find proc of origin, size, and indices
    rdispls[0]   = 0;
    n_recvs      = 0;
    int byte_ctr = 0;
    while (byte_ctr < local_size_msgs)
    {
        MPI_Unpack(local_recv_buffer.data(),
                   local_recv_buffer.size(),
                   &byte_ctr,
                   &(src[n_recvs]),
                   1,
                   MPI_INT,
                   comm->local_comm);
        MPI_Unpack(local_recv_buffer.data(),
                   local_recv_buffer.size(),
                   &byte_ctr,
                   &count,
                   1,
                   MPI_INT,
                   comm->local_comm);
        count = count / recv_bytes;

        MPI_Unpack(local_recv_buffer.data(),
                   local_recv_buffer.size(),
                   &byte_ctr,
                   &(recv_buffer[rdispls[n_recvs] * recv_bytes]),
                   count,
                   recvtype,
                   comm->local_comm);

        recvcounts[n_recvs]  = count;
        rdispls[n_recvs + 1] = rdispls[n_recvs] + count;
        n_recvs++;
    }
    // Set send sizes
    *recv_nnz  = n_recvs;
    *recv_size = rdispls[n_recvs];

    return MPI_SUCCESS;
}
