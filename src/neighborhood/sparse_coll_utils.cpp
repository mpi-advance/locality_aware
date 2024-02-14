#include "sparse_coll.h"

/* Assumes SMP Ordering of ranks across nodes (aggregates ranks 0-PPN) */
int alltoall_crs_nonblocking_loc(int send_nnz, int* dest, int sendcount, 
        MPI_Datatype sendtype, void* sendvals,
        int* recv_nnz, int* src, int recvcount, MPI_Datatype recvtype,
        void* recvvals, MPIX_Comm comm)
{ 
    int rank, num_procs, local_rank, PPN;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);
    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &PPN);

    char* send_buffer = (char*)sendvals;
    char* recv_buffer = (char*)recvvals;
    int send_bytes, recv_bytes;
    int MPI_Type_size(sendtype, &send_bytes);
    int MPI_Type_size(recvtype, &recv_bytes);

    if (comm->local_comm == MPI_COMM_NULL)
        MPIX_Comm_topo_init(comm);
    if (comm->n_requests < send_nnz)
        MPIX_Comm_req_resize(send_nnz);

    int tag = 790382;
    MPI_Status recv_status;
    MPI_Request bar_req;
    int proc, ctr, flag, ibar;

    char* node_send_buffer = (char*)malloc(send_nnz*(sendcount*send_bytes + sizeof(int)));

    // Send a message to every process that I will need data from
    // Tell them which global indices I need from them
    int node = -1;
    if (send_nnz > 0)
    {
        node = dest[0] / PPN;
    }

    first = 0;
    last = 0;
    n_sends = 0;
    for (int i = 0; i < send_nnz; i++)
    {
        proc = dest[i];
        if (proc / PPN != node)
        {
            MPI_Issend(&(node_send_buffer[first*send_bytes]), (last - first) * send_bytes, MPI_BYTE,
                    node*PPN + local_rank, tag, comm->global_comm, comm->requests[n_sends++]);
            first = last;
            node = proc / PPN;
        }
        memcpy(&(node_send_buffer[last]), &proc, sizeof(int));
        last += sizeof(int);
        memcpy(&(node_send_buffer[last]), &(send_buffer[i*send_bytes*sendcount]), send_bytes*sendcount);
        last += sendbytes*sendcount;
    }
    if (node >= 0)
    {
        MPI_Issend(&(node_send_buffer[first*send_bytes]), (last - first) * send_bytes, MPI_BYTE,
                node*PPN + local_rank, tag, comm->global_comm, comm->requests[n_sends++]);
    }


    std::vector<char> recv_buf;
    std::vector<int> origins;
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
            MPI_Recv(recv_buf[ctr], count, MPI_BYTE, proc, tag, comm->global_comm, 
                    &recv_status);
            ctr += count;

            n_msgs = count / (recvcount*recv_bytes + sizeof(int));
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
        ctr += (recvcount*recv_bytes + sizeof(int));
        msg_counts[proc]++;
    }

    std::vector<int> displs(PPN+1);
    displs[0] = 0;
    for (int i = 0; i < PPN; i++)
        displs[i+1] = displs[i] + (msg_counts[i]*(sendcount*send_bytes+sizeof(int)));

    std::vector<char> local_send_buffer(recv_buf.size());
    ctr = 0;
    idx = 0;
    while (ctr < recv_buf.size())
    {
        memcpy(&proc, &(recv_buf[ctr]), sizeof(int));
        ctr += sizeof(int);
        memcpy(&local_send_buffer[displs[proc]]), &origins[idx++], sizeof(int));
        displs[proc] += sizeof(int);
        memcpy(&local_send_buffer[displs[proc]]), &recv_buf[ctr]), recvcount*recv_bytes);
        ctr += recvcount*recv_bytes;
        displs[proc] += recvcount*recv_bytes;
    }
    displs[0] = 0;
    for (int i = 0; i < PPN; i++)
        displs[i+1] = displs[i] + (msg_counts[i]*(sendcount*send_bytes+sizeof(int)));

    MPI_Allreduce(MPI_IN_PLACE, msg_counts.data(), PPN, MPI_INT, MPI_SUM, comm->local_comm);
    recv_count = msg_counts[local_rank];
    if (PPN > comm->n_requests)
        MPIX_comm_req_resize(PPN);

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

    std::vector<char> local_recv_buffer(recv_count*(recvcount*recv_bytes + sizeof(int));

    // Wait to receive values
    // until I have received fewer than the number of global indices I am waiting on
    ctr = 0;
    for (int i = 0; i < recv_count; i++)
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
        memcpy(&(recv_buffer[new_idx]), &(local_recv_buffer[idx]), recvcount*recv_bytes);
        idx += recvcount*recv_bytes;
        new_idx += recvcount*recv_bytes;
    }
    *recv_nnz = n_recvs;

}

