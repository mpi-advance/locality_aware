#include <cstring>
#include <vector>

#include "sparse_coll.h"

int alltoallv_crs_personalized(const int send_nnz,
                               const int send_size,
                               const int* dest,
                               const int* sendcounts,
                               const int* sdispls,
                               MPI_Datatype sendtype,
                               const void* sendvals,
                               int* recv_nnz,
                               int* recv_size,
                               int** src_ptr,
                               int** recvcounts_ptr,
                               int** rdispls_ptr,
                               MPI_Datatype recvtype,
                               void** recvvals_ptr,
                               MPIL_Info* xinfo,
                               MPIL_Comm* comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    MPI_Status recv_status;
    int proc, ctr, count;
    int tag;
    MPIL_Comm_tag(comm, &tag);

    char* send_buffer = (char*)sendvals;
    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    // Allreduce to determine recv_size
    // TODO - Could use reduce scatter (useful if large)
    std::vector<int> msg_counts(num_procs, 0);
    for (int i = 0; i < send_nnz; i++)
    {
        msg_counts[dest[i]] = sendcounts[i] * send_bytes;
    }
    MPI_Allreduce(
        MPI_IN_PLACE, msg_counts.data(), num_procs, MPI_INT, MPI_SUM, comm->global_comm);
    *recv_size = msg_counts[rank] / recv_bytes;

    // Allocate recvvals to size determined in allreduce
    char* recvvals;
    MPIL_Alloc((void**)&recvvals, *recv_size * recv_bytes);

    if (comm->n_requests < send_nnz)
    {
        MPIL_Comm_req_resize(comm, send_nnz);
    }

    // Send each message
    for (int i = 0; i < send_nnz; i++)
    {
        proc = dest[i];
        MPI_Isend(&(send_buffer[sdispls[i] * send_bytes]),
                  sendcounts[i] * send_bytes,
                  MPI_BYTE,
                  proc,
                  tag,
                  comm->global_comm,
                  &(comm->requests[i]));
    }

    ctr = 0;
    std::vector<int> src;
    std::vector<int> recvcounts;
    std::vector<int> rdispls;
    rdispls.push_back(ctr);
    while (ctr < *recv_size)
    {
        // Find a message
        MPI_Probe(MPI_ANY_SOURCE, tag, comm->global_comm, &recv_status);

        // Push back the process
        proc = recv_status.MPI_SOURCE;
        src.push_back(proc);

        MPI_Get_count(&recv_status, MPI_BYTE, &count);
        count /= recv_bytes;
        recvcounts.push_back(count);

        MPI_Recv(&(recvvals[ctr * recv_bytes]),
                 count * recv_bytes,
                 MPI_BYTE,
                 proc,
                 tag,
                 comm->global_comm,
                 &recv_status);

        ctr += count;
        rdispls.push_back(ctr);
    }
    *recv_nnz     = src.size();
    *recvvals_ptr = recvvals;

    MPIL_Alloc((void**)src_ptr, src.size() * sizeof(int));
    MPIL_Alloc((void**)recvcounts_ptr, recvcounts.size() * sizeof(int));
    MPIL_Alloc((void**)rdispls_ptr, rdispls.size() * sizeof(int));
    memcpy((*src_ptr), src.data(), src.size() * sizeof(int));
    memcpy((*recvcounts_ptr), recvcounts.data(), recvcounts.size() * sizeof(int));
    memcpy((*rdispls_ptr), rdispls.data(), rdispls.size() * sizeof(int));

    if (send_nnz)
    {
        MPI_Waitall(send_nnz, comm->requests, MPI_STATUSES_IGNORE);
    }

    return MPI_SUCCESS;
}

int alltoallv_crs_nonblocking(const int send_nnz,
                              const int send_size,
                              const int* dest,
                              const int* sendcounts,
                              const int* sdispls,
                              MPI_Datatype sendtype,
                              const void* sendvals,
                              int* recv_nnz,
                              int* recv_size,
                              int** src_ptr,
                              int** recvcounts_ptr,
                              int** rdispls_ptr,
                              MPI_Datatype recvtype,
                              void** recvvals_ptr,
                              MPIL_Info* xinfo,
                              MPIL_Comm* comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    char* send_buffer = (char*)sendvals;
    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    int proc, ctr, flag, ibar, count;
    MPI_Status recv_status;
    MPI_Request bar_req;
    int tag;
    MPIL_Comm_tag(comm, &tag);

    if (comm->n_requests < send_nnz)
    {
        MPIL_Comm_req_resize(comm, send_nnz);
    }

    for (int i = 0; i < send_nnz; i++)
    {
        proc = dest[i];
        MPI_Issend(&(send_buffer[sdispls[i] * send_bytes]),
                   sendcounts[i] * send_bytes,
                   MPI_BYTE,
                   proc,
                   tag,
                   comm->global_comm,
                   &(comm->requests[i]));
    }

    ibar = 0;
    ctr  = 0;
    std::vector<int> src;
    std::vector<int> recvcounts;
    std::vector<int> rdispls;
    std::vector<char> recvvals;
    rdispls.push_back(0);
    while (1)
    {
        MPI_Iprobe(MPI_ANY_SOURCE, tag, comm->global_comm, &flag, &recv_status);
        if (flag)
        {
            MPI_Probe(MPI_ANY_SOURCE, tag, comm->global_comm, &recv_status);

            proc = recv_status.MPI_SOURCE;
            src.push_back(proc);

            MPI_Get_count(&recv_status, MPI_BYTE, &count);
            recvvals.resize(recvvals.size() + count);

            count /= recv_bytes;
            recvcounts.push_back(count);

            MPI_Recv(&(recvvals[ctr * recv_bytes]),
                     count * recv_bytes,
                     MPI_BYTE,
                     proc,
                     tag,
                     comm->global_comm,
                     &recv_status);

            ctr += count;
            rdispls.push_back(ctr);
        }
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
            MPI_Testall(send_nnz, comm->requests, &flag, MPI_STATUSES_IGNORE);
            if (flag)
            {
                ibar = 1;
                MPI_Ibarrier(comm->global_comm, &bar_req);
            }
        }
    }
    *recv_nnz  = src.size();
    *recv_size = ctr;

    MPIL_Alloc((void**)src_ptr, src.size() * sizeof(int));
    MPIL_Alloc((void**)recvcounts_ptr, recvcounts.size() * sizeof(int));
    MPIL_Alloc((void**)rdispls_ptr, rdispls.size() * sizeof(int));
    MPIL_Alloc((void**)recvvals_ptr, recvvals.size());
    //(*src_ptr) = (int*)MPIalloc(src.size()*sizeof(int));
    //(*recvcounts_ptr) = (int*)MPIalloc(recvcounts.size()*sizeof(int));
    //(*rdispls_ptr) = (int*)MPIalloc(rdispls.size()*sizeof(int));
    //(*recvvals_ptr) = MPIalloc(recvvals.size());

    memcpy((*src_ptr), src.data(), src.size() * sizeof(int));
    memcpy((*recvcounts_ptr), recvcounts.data(), recvcounts.size() * sizeof(int));
    memcpy((*rdispls_ptr), rdispls.data(), rdispls.size() * sizeof(int));
    memcpy((*recvvals_ptr), recvvals.data(), recvvals.size());

    return MPI_SUCCESS;
}

void local_redistribute(int n_recvs,
                        std::vector<int>& origins,
                        std::vector<int>& origin_displs,
                        std::vector<char>& recv_buf,
                        int* recv_nnz,
                        int* recv_size,
                        int** src_ptr,
                        int** recvcounts_ptr,
                        int** rdispls_ptr,
                        MPI_Datatype recvtype,
                        void** recvvals_ptr,
                        MPIL_Info* xinfo,
                        MPIL_Comm* comm)
{
    int rank, num_procs, local_rank, PPN;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);
    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &PPN);

    MPI_Status recv_status;
    int count, ctr, tag, proc, n_sends;
    int recv_bytes, int_bytes;
    MPI_Type_size(recvtype, &recv_bytes);
    MPI_Type_size(MPI_INT, &int_bytes);

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

    MPIL_Comm_tag(comm, &tag);

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
    int byte_ctr = 0;
    std::vector<int> src;
    std::vector<int> recvcounts;
    std::vector<int> rdispls;
    std::vector<char> recvvals;
    rdispls.push_back(0);
    ctr = 0;
    while (byte_ctr < local_size_msgs)
    {
        MPI_Unpack(local_recv_buffer.data(),
                   local_recv_buffer.size(),
                   &byte_ctr,
                   &proc,
                   1,
                   MPI_INT,
                   comm->local_comm);
        src.push_back(proc);

        MPI_Unpack(local_recv_buffer.data(),
                   local_recv_buffer.size(),
                   &byte_ctr,
                   &count,
                   1,
                   MPI_INT,
                   comm->local_comm);

        recvvals.resize(recvvals.size() + count);
        count = count / recv_bytes;
        recvcounts.push_back(count);

        MPI_Unpack(local_recv_buffer.data(),
                   local_recv_buffer.size(),
                   &byte_ctr,
                   &(recvvals[ctr * recv_bytes]),
                   count,
                   recvtype,
                   comm->local_comm);

        ctr += count;
        rdispls.push_back(ctr);
    }
    // Set send sizes
    *recv_nnz  = src.size();
    *recv_size = ctr;

    MPIL_Alloc((void**)src_ptr, src.size() * sizeof(int));
    MPIL_Alloc((void**)recvcounts_ptr, recvcounts.size() * sizeof(int));
    MPIL_Alloc((void**)rdispls_ptr, rdispls.size() * sizeof(int));
    MPIL_Alloc(recvvals_ptr, recvvals.size());
    //(*src_ptr) = (int*)MPIalloc(src.size()*sizeof(int));
    //(*recvcounts_ptr) = (int*)MPIalloc(recvcounts.size()*sizeof(int));
    //(*rdispls_ptr) = (int*)MPIalloc(rdispls.size()*sizeof(int));
    //(*recvvals_ptr) = MPIalloc(recvvals.size());

    memcpy((*src_ptr), src.data(), src.size() * sizeof(int));
    memcpy((*recvcounts_ptr), recvcounts.data(), recvcounts.size() * sizeof(int));
    memcpy((*rdispls_ptr), rdispls.data(), rdispls.size() * sizeof(int));
    memcpy((*recvvals_ptr), recvvals.data(), recvvals.size());
}

int alltoallv_crs_personalized_loc(const int send_nnz,
                                   const int send_size,
                                   const int* dest,
                                   const int* sendcounts,
                                   const int* sdispls,
                                   MPI_Datatype sendtype,
                                   const void* sendvals,
                                   int* recv_nnz,
                                   int* recv_size,
                                   int** src_ptr,
                                   int** recvcounts_ptr,
                                   int** rdispls_ptr,
                                   MPI_Datatype recvtype,
                                   void** recvvals_ptr,
                                   MPIL_Info* xinfo,
                                   MPIL_Comm* comm)
{
    int rank, num_procs, local_rank, PPN;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    if (comm->local_comm == MPI_COMM_NULL)
    {
        MPIL_Comm_topo_init(comm);
    }
    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &PPN);

    if (comm->n_requests < send_nnz)
    {
        MPIL_Comm_req_resize(comm, send_nnz);
    }

    int tag;
    MPIL_Comm_tag(comm, &tag);

    char* send_buffer = (char*)sendvals;
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

    local_redistribute(n_recvs,
                       origins,
                       origin_displs,
                       recv_buf,
                       recv_nnz,
                       recv_size,
                       src_ptr,
                       recvcounts_ptr,
                       rdispls_ptr,
                       recvtype,
                       recvvals_ptr,
                       xinfo,
                       comm);

    return MPI_SUCCESS;
}

int alltoallv_crs_nonblocking_loc(const int send_nnz,
                                  const int send_size,
                                  const int* dest,
                                  const int* sendcounts,
                                  const int* sdispls,
                                  MPI_Datatype sendtype,
                                  const void* sendvals,
                                  int* recv_nnz,
                                  int* recv_size,
                                  int** src_ptr,
                                  int** recvcounts_ptr,
                                  int** rdispls_ptr,
                                  MPI_Datatype recvtype,
                                  void** recvvals_ptr,
                                  MPIL_Info* xinfo,
                                  MPIL_Comm* comm)
{
    int rank, num_procs, local_rank, PPN;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    if (comm->local_comm == MPI_COMM_NULL)
    {
        MPIL_Comm_topo_init(comm);
    }
    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &PPN);

    if (comm->n_requests < send_nnz)
    {
        MPIL_Comm_req_resize(comm, send_nnz);
    }

    int tag;
    MPIL_Comm_tag(comm, &tag);

    char* send_buffer = (char*)sendvals;
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

    local_redistribute(n_recvs,
                       origins,
                       origin_displs,
                       recv_buf,
                       recv_nnz,
                       recv_size,
                       src_ptr,
                       recvcounts_ptr,
                       rdispls_ptr,
                       recvtype,
                       recvvals_ptr,
                       xinfo,
                       comm);

    return MPI_SUCCESS;
}
