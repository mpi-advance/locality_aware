#include "collective.h"

/**************************************************
 * Locality-Aware Point-to-Point Alltoallv
 * Same as PMPI_Alltoall (no load balancing)
 *  - Aggregates messages locally to reduce 
 *      non-local communciation
 *  - First redistributes on-node so that each
 *      process holds all data for a subset
 *      of other nodes
 *  - Then, performs inter-node communication
 *      during which each process exchanges
 *      data with their assigned subset of nodes
 *  - Finally, redistribute received data
 *      on-node so that each process holds
 *      the correct final data
 *  - To be used when sizes are relatively balanced
 *  - For load balacing, use persistent version
 *      - Load balacing is too expensive for 
 *          non-persistent Alltoallv
 *************************************************/
int MPI_Alltoallv(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPI_Comm comm)
{
    MPIX_Comm* locality_comm;
    MPIX_Comm_init(&locality_comm, comm);

    MPIX_Alltoallv(sendbuf, sendcounts, sdispls, sendtype,
            recvbuf, recvcounts, rdispls, recvtype,
            locality_comm);

    MPIX_Comm_free(locality_comm);
}

int MPIX_Alltoallv(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* mpi_comm)
{
    int rank, num_procs;
    MPI_Comm_rank(mpi_comm->global_comm, &rank);
    MPI_Comm_size(mpi_comm->global_comm, &num_procs);

    // Create shared-memory (local) communicator
    int local_rank, PPN;
    MPI_Comm_rank(mpi_comm->local_comm, &local_rank);
    MPI_Comm_size(mpi_comm->local_comm, &PPN);

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
    long start, end;
    long ctr, next_ctr;

    int send_ctr, recv_ctr;
    int node_size, size;

    int num_msgs = num_nodes / PPN; // TODO : this includes talking to self
    int extra = num_nodes % PPN;
    int local_num_msgs = num_msgs;
    if (local_rank < extra) local_num_msgs++;

    int* proc_node_sizes = (int*)malloc(PPN*sizeof(int));
    int* proc_node_displs = (int*)malloc((PPN+1)*sizeof(int));
    int* local_node_sizes = (int*)malloc(PPN*sizeof(int));
    int* local_node_displs = (int*)malloc((PPN+1)*sizeof(int));

    int* local_S_send_displs = (int*)malloc((PPN+1)*sizeof(int));
    int* local_R_recv_displs = (int*)malloc((PPN+1)*sizeof(int));
    long* local_S_recv_displs = (long*)malloc((PPN+1)*sizeof(long));
    long* local_R_send_displs = (long*)malloc((PPN+1)*sizeof(long));

    proc_node_displs[0] = 0;
    local_node_displs[0] = 0;
    for (int i = 0; i < PPN; i++)
    {
        proc_node_sizes[i] = num_msgs;
        if (i < extra) proc_node_sizes[i]++;
        proc_node_displs[i+1] = proc_node_displs[i] + proc_node_sizes[i];
        if (proc_node_displs[i] <= local_node && proc_node_displs[i+1] > local_node)
            local_idx = i;

        local_node_sizes[i] = local_num_msgs;
        local_node_displs[i+1] = local_node_displs[i] + local_node_sizes[i];
    }
    int first_msg = proc_node_displs[local_rank];

    int* orig_node_sizes = (int*)malloc(num_nodes*sizeof(int));
    local_S_send_displs[0] = 0;
    local_R_recv_displs[0] = 0;
    // final_proc_sizes = recvcounts!
    int s_size, r_size;
    send_ctr = 0;
    recv_ctr = 0;
    for (int i = 0; i < PPN; i++)
    {
        start = proc_node_displs[i];
        end = proc_node_displs[i+1];
        s_size = 0;
        r_size = 0;
        for (int node = start; node < end; node++)
        {
            node_size = 0;
            for (int j = 0; j < PPN; j++)
            {
                proc = get_global_proc(mpi_comm, node, j);
                s_size += sendcounts[proc];
                r_size += recvcounts[proc];
                node_size += sendcounts[proc];
            }
            orig_node_sizes[node] = node_size;
        }
        send_ctr += s_size;
        local_S_send_displs[i+1] = send_ctr;
        recv_ctr += r_size;
        local_R_recv_displs[i+1] = recv_ctr;
    }

    int* send_node_sizes = (int*)malloc((local_num_msgs*PPN)*sizeof(int));
    PMPI_Alltoallv(orig_node_sizes,
        proc_node_sizes,
        proc_node_displs,
        MPI_INT,
        send_node_sizes,
        local_node_sizes,
        local_node_displs,
        MPI_INT,
        mpi_comm->local_comm);
    free(orig_node_sizes);

    local_S_recv_displs[0] = 0;
    for (int i = 0; i < PPN; i++)
    {
        size = 0;
        for (int j = 0; j < local_num_msgs; j++)
        {
            size += send_node_sizes[i*local_num_msgs+j];
        }
        local_S_recv_displs[i+1] = local_S_recv_displs[i] + size;
    }

    for (int i = 0; i < PPN; i++)
    {
        proc_node_sizes[i] *= PPN;
        proc_node_displs[i] *= PPN;
        local_node_sizes[i] *= PPN;
        local_node_displs[i] *= PPN;
    }

    int* recv_proc_sizes = (int*)malloc((local_num_msgs*PPN*PPN)*sizeof(int));
    PMPI_Alltoallv(recvcounts,
        proc_node_sizes,
        proc_node_displs,
        MPI_INT,
        recv_proc_sizes,
        local_node_sizes,
        local_node_displs,
        MPI_INT,
        mpi_comm->local_comm);

    local_R_send_displs[0] = 0;
    for (int i = 0; i < PPN; i++)
    {
        size = 0;
        for (int j = 0; j < local_num_msgs; j++)
        {
            for (int k = 0; k < PPN; k++)
            {
                size += recv_proc_sizes[i*local_num_msgs*PPN + j*PPN + k];
            }
        }
        local_R_send_displs[i+1] = local_R_send_displs[i] + size;
    }

    int idx;
    int* recv_proc_displs = (int*)malloc((local_num_msgs*PPN*PPN+1)*sizeof(int));
    recv_proc_displs[0] = 0;
    for (int i = 0; i < local_num_msgs; i++)
    {
        for (int j = 0; j < PPN; j++)
        {
            for (int k = 0; k < PPN; k++)
            {
                idx = j*local_num_msgs*PPN + i*PPN + k;
                recv_proc_displs[idx+1] = recv_proc_displs[idx] + recv_proc_sizes[idx];
            }
        }
    }

    free(local_node_sizes);
    free(local_node_displs);
    free(proc_node_sizes);
    free(proc_node_displs);

    int n_msgs;

    long buf_size = local_S_recv_displs[PPN];
    if (local_R_send_displs[PPN] > buf_size)
        buf_size = local_R_send_displs[PPN];
    buf_size *= recv_size;

    char* tmpbuf = NULL;
    char* contig_buf = NULL;

    if (buf_size)
    {
        tmpbuf = (char*)malloc(buf_size*sizeof(char));
        contig_buf = (char*)malloc(buf_size*sizeof(char));
    }

    MPI_Request* local_requests = (MPI_Request*)malloc(2*PPN*sizeof(MPI_Request));
    MPI_Request* nonlocal_requests = (MPI_Request*)malloc(2*local_num_msgs*sizeof(MPI_Request));

     /************************************************
     * Step 1 : local Alltoall
     *      Redistribute data so that local rank x holds
     *      all data that needs to be send to any
     *      node with which local rank x communicates
     ************************************************/
    // Alltoall to distribute message sizes 
    n_msgs = 0;
    ctr = 0;
    for (int i = 0; i < PPN; i++)
    {
        start = local_S_send_displs[i];
        end = local_S_send_displs[i+1];
        if (end - start)
        {
            MPI_Isend(&(send_buffer[ctr*send_size]), 
                    end - start, 
                    sendtype, 
                    i, 
                    tag, 
                    mpi_comm->local_comm, 
                    &(local_requests[n_msgs++]));
        }
        ctr += (end - start);
    }

    ctr = 0;
    for (int i = 0; i < PPN; i++)
    {
        start = local_S_recv_displs[i];
        end = local_S_recv_displs[i+1];
        if (end - start)
        {
            MPI_Irecv(&(tmpbuf[ctr*send_size]), 
                    end - start, 
                    sendtype,
                    i, 
                    tag, 
                    mpi_comm->local_comm, 
                    &(local_requests[n_msgs++]));
        }
        ctr += (end - start);
    }
    if (n_msgs)
        MPI_Waitall(n_msgs, local_requests, MPI_STATUSES_IGNORE);

     /************************************************
     * Step 2 : non-local Alltoall
     *      Local rank x exchanges data with 
     *      local rank x on nodes x, PPN+x, 2PPN+x, etc
     ************************************************/

    n_msgs = 0;
    ctr = 0;
    next_ctr = ctr;
    for (int i = 0; i < local_num_msgs; i++)
    {
        node = first_msg + i;
        proc = node*PPN + local_idx;
        for (int j = 0; j < PPN; j++)
        {
            start = local_S_recv_displs[j];
            size = send_node_sizes[j*local_num_msgs + i];

            for (int k = 0; k < size; k++)
            {
                for (int l = 0; l < send_size; l++)
                {
                    contig_buf[next_ctr*send_size+l] = tmpbuf[(start+k)*send_size + l];
                }
                next_ctr++;
            }
            local_S_recv_displs[j] = start + size;
        }
        if (next_ctr - ctr)
        {

            MPI_Isend(&(contig_buf[ctr*send_size]), 
                    next_ctr - ctr,
                    sendtype, 
                    proc, 
                    tag, 
                    mpi_comm->global_comm, 
                    &(nonlocal_requests[n_msgs++]));

        }
        ctr = next_ctr;
    }


    ctr = 0;
    for (int i = 0; i < local_num_msgs; i++)
    {
        node = first_msg + i;
        proc = node*PPN + local_idx;
        start = recv_proc_displs[i*PPN*PPN];
        end = recv_proc_displs[(i+1)*PPN*PPN];
        if (end - start)
        {
            MPI_Irecv(&(tmpbuf[ctr*recv_size]), 
                    end - start, 
                    recvtype,
                    proc, 
                    tag,
                    mpi_comm->global_comm, 
                    &(nonlocal_requests[n_msgs++]));
        }
        ctr += (end - start);
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
                start = recv_proc_displs[j*PPN*PPN + k*PPN + i];
                size = recv_proc_sizes[j*PPN*PPN + k*PPN + i];
                for (int l = 0; l < size; l++)
                {
                    for (int m = 0; m < recv_size; m++)
                    {
                        contig_buf[next_ctr*recv_size+m] = tmpbuf[(start+l)*recv_size+m];
                    }
                    next_ctr++;
                }
            }
        }

        if (next_ctr - ctr)
        {
            MPI_Isend(&(contig_buf[ctr*recv_size]), 
                    next_ctr - ctr,
                    recvtype,
                    i,
                    local_tag,
                    mpi_comm->local_comm,
                    &(local_requests[n_msgs++]));
        }
        ctr = next_ctr;
    }

    ctr = 0;
    for (int i = 0; i < PPN; i++)
    {
        start = local_R_recv_displs[i];
        end = local_R_recv_displs[i+1];
        if (end - start)
        {
            MPI_Irecv(&(recv_buffer[ctr*recv_size]), 
                    (end - start)*recv_size,
                    recvtype, 
                    i, 
                    local_tag, 
                    mpi_comm->local_comm, 
                    &(local_requests[n_msgs++]));
        }
        ctr += (end - start);
    }

    MPI_Waitall(n_msgs,
            local_requests, 
            MPI_STATUSES_IGNORE);
            

    if (buf_size)
    {
        free(tmpbuf);
        free(contig_buf);
    }

    free(local_requests);
    free(nonlocal_requests);

    free(local_S_send_displs);
    free(local_S_recv_displs);
    free(local_R_send_displs);
    free(local_R_recv_displs);

    free(send_node_sizes);
    free(recv_proc_sizes);
    free(recv_proc_displs);

    return 0;
}
