#include "collective.h"

// TODO : Add Locality-Aware Bruck Alltoall Algorithm!


int PMPI_Alltoall(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    // Create shared-memory (local) communicator
    MPI_Comm local_comm;
    int local_rank, PPN;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &PPN);

    // Calculate shared-memory (local) variables
    int num_nodes = num_procs / PPN;
    int local_node = rank / PPN;

    // Local rank x sends to nodes [x:x/PPN] etc
    // Which local rank from these nodes sends to my node?
    // E.g. if local_rank 0 sends to nodes 0,1,2 and 
    // my rank is on node 2, I want to receive from local_rank 0
    // regardless of the node I talk to
    int local_idx = -1;

    const char* send_buffer = (char*) sendbuf;
    char* recv_buffer = (char*) recvbuf;
    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    /************************************************
     * Setup : determine message sizes and displs for
     *      intermediate (aggregated) steps
     ************************************************/
    int proc, node;
    int tag = 923812;
    int local_tag = 728401;
    int start, end;
    int ctr, next_ctr;

    int num_msgs = num_nodes / PPN; // TODO : this includes talking to self
    int extra = num_nodes % PPN;
    int local_num_msgs = num_msgs;
    if (local_rank < extra) local_num_msgs++;

    int send_msg_size = sendcount*PPN;
    int recv_msg_size = recvcount*PPN;
    int* local_send_displs = (int*)malloc((PPN+1)*sizeof(int));
    local_send_displs[0] = 0;
    for (int i = 0; i < PPN; i++)
    {
        ctr = num_msgs;
        if (i < extra) ctr++;
        local_send_displs[i+1] = local_send_displs[i] + ctr;
        if (local_send_displs[i] <= local_node && local_send_displs[i+1] > local_node)
            local_idx = i;
    }
    int first_msg = local_send_displs[local_rank];
    int n_msgs;

    int bufsize = (num_msgs+1)*recv_msg_size*PPN*recv_size;
    char* tmpbuf = (char*)malloc(bufsize*sizeof(char));
    char* contig_buf = (char*)malloc(bufsize*sizeof(char));
    MPI_Request* local_requests = (MPI_Request*)malloc(2*PPN*sizeof(MPI_Request));
    MPI_Request* nonlocal_requests = (MPI_Request*)malloc(2*local_num_msgs*sizeof(MPI_Request));

     /************************************************
     * Step 1 : local Alltoall
     *      Redistribute data so that local rank x holds
     *      all data that needs to be send to any
     *      node with which local rank x communicates
     ************************************************/
    n_msgs = 0;
    for (int i = 0; i < PPN; i++)
    {
        start = local_send_displs[i];
        end = local_send_displs[i+1];
        if (end - start)
        {
            MPI_Isend(&(send_buffer[start*send_msg_size*send_size]), 
                    (end - start)*send_msg_size, 
                    sendtype, 
                    i, 
                    tag, 
                    local_comm, 
                    &(local_requests[n_msgs++]));
        }
        if (local_num_msgs)
        {
            MPI_Irecv(&(tmpbuf[i*local_num_msgs*recv_msg_size*recv_size]), 
                    local_num_msgs*recv_msg_size, 
                    recvtype,
                    i, 
                    tag, 
                    local_comm, 
                    &(local_requests[n_msgs++]));
        }
    }
    if (n_msgs)
        MPI_Waitall(n_msgs, local_requests, MPI_STATUSES_IGNORE);

     /************************************************
     * Step 2 : non-local Alltoall
     *      Local rank x exchanges data with 
     *      local rank x on nodes x, PPN+x, 2PPN+x, etc
     ************************************************/
    ctr = 0;
    next_ctr = ctr;
    n_msgs = 0;
    for (int i = 0; i < local_num_msgs; i++)
    {
        node = first_msg + i;
        proc = node*PPN + local_idx;
        for (int j = 0; j < PPN; j++)
        {
            for (int k = 0; k < recv_msg_size; k++)
            {
                for (int l = 0; l < recv_size; l++)
                {
                    contig_buf[next_ctr*recv_size+l] = tmpbuf[(i*recv_msg_size +
                            j*recv_msg_size*local_num_msgs + k)*recv_size + l];
                }
                next_ctr++;
            }
        }
        if (next_ctr - ctr)
        {
            MPI_Isend(&(contig_buf[ctr*recv_size]), 
                    next_ctr - ctr,
                    recvtype, 
                    proc, 
                    tag, 
                    comm, 
                    &(nonlocal_requests[n_msgs++]));
            MPI_Irecv(&(tmpbuf[ctr*recv_size]), 
                    next_ctr - ctr, 
                    recvtype, 
                    proc, 
                    tag,
                    comm, 
                    &(nonlocal_requests[n_msgs++]));
        }
        ctr = next_ctr;
    }
    if (n_msgs) 
        MPI_Waitall(n_msgs,
                nonlocal_requests,
                MPI_STATUSES_IGNORE);

     /************************************************
     * Step 3 : local Alltoall
     *      Locally redistribute all received data
     ************************************************/
    ctr = 0;
    next_ctr = ctr;
    n_msgs = 0;
    for (int i = 0; i < PPN; i++)
    {
        for (int j = 0; j < local_num_msgs; j++)
        {
            for (int k = 0; k < PPN; k++)
            {
                for (int l = 0; l < recvcount; l++)
                {
                    for (int m = 0; m < recv_size; m++)
                    {
                        contig_buf[next_ctr*recv_size+m] = tmpbuf[((((j*PPN+k)*PPN+i)*recvcount)+l)*recv_size+m];
                    }
                    next_ctr++;
                }
            }
        }
        start = local_send_displs[i];
        end = local_send_displs[i+1];

        if (next_ctr - ctr)
        {
            MPI_Isend(&(contig_buf[ctr*recv_size]), 
                    next_ctr - ctr,
                    recvtype,
                    i,
                    local_tag,
                    local_comm,
                    &(local_requests[n_msgs++]));
        }

        if (end - start)
        {
            MPI_Irecv(&(recv_buffer[(start*PPN*recvcount)*recv_size]), 
                    (end - start)*PPN*recvcount*recv_size,
                    recvtype, 
                    i, 
                    local_tag, 
                    local_comm, 
                    &(local_requests[n_msgs++]));
        }
        ctr = next_ctr;
    }
    MPI_Waitall(n_msgs,
            local_requests, 
            MPI_STATUSES_IGNORE);
            

    MPI_Comm_free(&local_comm);

    free(local_send_displs);
    free(tmpbuf);
    free(contig_buf);
    free(local_requests);
    free(nonlocal_requests);
}
