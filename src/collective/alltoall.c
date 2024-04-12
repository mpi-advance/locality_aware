#include "alltoall.h"
#include <string.h>
#include <math.h>
#include "utils.h"


// TODO : Add Locality-Aware Bruck Alltoall Algorithm!
// TODO : Change to PMPI_Alltoall and test with profiling library!

/**************************************************
 * Locality-Aware Point-to-Point Alltoall
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
 *************************************************/
int MPI_Alltoall(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm)
{
    return alltoall_pairwise(sendbuf,
        sendcount,
        sendtype,
        recvbuf,
        recvcount,
        recvtype,
        comm);
}


int MPIX_Alltoall(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* mpi_comm)
{    
    return alltoall_pairwise_loc(sendbuf,
        sendcount,
        sendtype,
        recvbuf,
        recvcount,
        recvtype,
        mpi_comm);
}

int alltoall_rma(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf, 
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* xcomm)
{
    int rank, num_procs;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_size(xcomm->global_comm, &num_procs);

    char* send_buffer = (char*)(sendbuf);
    char* recv_buffer = (char*)(recvbuf);

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);
    int bytes = num_procs * recvcount * recv_bytes;

    if (xcomm->win_bytes != bytes
            || xcomm->win_type_bytes != 1)
        MPIX_Comm_win_free(xcomm);

    if (xcomm->win == MPI_WIN_NULL)
    {
        MPIX_Comm_win_init(xcomm, bytes, 1);
    }

    send_bytes *= sendcount;
    recv_bytes *= recvcount;

    MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOPRECEDE, xcomm->win);
    for (int i = 0; i < num_procs; i++)
    {
         MPI_Put(&(send_buffer[i*send_bytes]), send_bytes, MPI_CHAR,
                 i, rank*recv_bytes, recv_bytes, MPI_CHAR, xcomm->win);
    }
    MPI_Win_fence(MPI_MODE_NOPUT|MPI_MODE_NOSUCCEED, xcomm->win);

    // Need to memcpy because win_array is created with window
    // TODO : could explore just attaching recv_buffer to existing dynamic window 
    //        with persistent collectives
    memcpy(recv_buffer, xcomm->win_array, bytes);

    return MPI_SUCCESS;
}

int alltoall_pairwise(const void* sendbuf,
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

    int tag = 102944;
    int send_proc, recv_proc;
    int send_pos, recv_pos;
    MPI_Status status;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    char* send_buffer = (char*)(sendbuf);
    char* recv_buffer = (char*)(recvbuf);

    // Send to rank + i
    // Recv from rank - i
    for (int i = 0; i < num_procs; i++)
    {
        send_proc = rank + i;
        if (send_proc >= num_procs)
            send_proc -= num_procs;
        recv_proc = rank - i;
        if (recv_proc < 0)
            recv_proc += num_procs;
        send_pos = send_proc * sendcount * send_size;
        recv_pos = recv_proc * recvcount * recv_size;

        MPI_Sendrecv(send_buffer + send_pos, sendcount, sendtype, send_proc, tag,
                recv_buffer + recv_pos, recvcount, recvtype, recv_proc, tag,
                comm, &status);
    }

    return MPI_SUCCESS;
}

int alltoall_init(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* xcomm,
        MPIX_Info* xinfo,
        MPIX_Request** request_ptr)
{
    int rank, num_procs;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_size(xcomm->global_comm, &num_procs);

    MPIX_Request* request;
    MPIX_Request_init(&request);
    request->global_n_msgs = 2*num_procs;
    allocate_requests(request->global_n_msgs, &(request->global_requests));

    request->start_function = batch_start;
    request->wait_function = batch_wait;

    int tag = 102944;
    int send_proc, recv_proc;
    int send_pos, recv_pos;
    MPI_Status status;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    char* send_buffer = (char*)(sendbuf);
    char* recv_buffer = (char*)(recvbuf);

    // Send to rank + i
    // Recv from rank - i
    for (int i = 0; i < num_procs; i++)
    {
        send_proc = rank + i;
        if (send_proc >= num_procs)
            send_proc -= num_procs;
        recv_proc = rank - i;
        if (recv_proc < 0)
            recv_proc += num_procs;
        send_pos = send_proc * sendcount * send_size;
        recv_pos = recv_proc * recvcount * recv_size;

        MPI_Send_init(send_buffer + send_pos, sendcount, sendtype, send_proc, tag,
                xcomm->global_comm, &(request->global_requests[2*i]));
        MPI_Recv_init(recv_buffer + recv_pos, recvcount, recvtype, recv_proc, tag,
                xcomm->global_comm, &(request->global_requests[2*i + 1]));
    }

    *request_ptr = request;
}

int alltoall_pairwise_init(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* xcomm,
        MPIX_Info* xinfo,
        MPIX_Request** request_ptr)
{
    
    alltoall_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, xcomm,
            xinfo, request_ptr);
    (*request_ptr)->batch = 1;

    return MPI_SUCCESS;
}

int alltoall_nonblocking_init(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* xcomm,
        MPIX_Info* xinfo,
        MPIX_Request** request_ptr)
{
    int num_procs;
    MPI_Comm_size(xcomm->global_comm, &num_procs);

    alltoall_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, xcomm,
            xinfo, request_ptr);
    (*request_ptr)->batch = num_procs;

    return MPI_SUCCESS;
}


// 2-Step Aggregation (large messages)
// Gather all data to be communicated between nodes
// Send to node+i, recv from node-i
int alltoall_pairwise_loc(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* mpi_comm)
{
    int rank, num_procs;
    int local_rank, PPN; 
    int num_nodes, rank_node;
    MPI_Comm_rank(mpi_comm->global_comm, &rank);
    MPI_Comm_size(mpi_comm->global_comm, &num_procs);
    MPI_Comm_rank(mpi_comm->local_comm, &local_rank);
    MPI_Comm_size(mpi_comm->local_comm, &PPN);
    num_nodes = mpi_comm->num_nodes;
    rank_node = mpi_comm->rank_node;

    char* send_buffer = (char*)(sendbuf);
    char* recv_buffer = (char*)(recvbuf);

    int sbytes, rbytes;
    MPI_Type_size(sendtype, &sbytes);
    MPI_Type_size(recvtype, &rbytes);
    int sendcount_node = sendcount * PPN;
    int recvcount_node = recvcount * PPN;
    int recv_bytes = recvcount*rbytes;
    int send_bytes_node = sendcount_node * sbytes;
    int recv_bytes_node = recvcount_node * rbytes;

    int tag = 102913;
    int send_proc, recv_proc;
    int send_pos, recv_pos;
    int send_node, recv_node;
    MPI_Status status;
    char* tmpbuf = (char*)malloc(num_procs*recv_bytes);

    /************************************************
     * Step 1 : Send aggregated data to node
     ***********************************************/
    for (int i = 0; i < num_nodes; i++)
    {
        send_node = rank_node + i;
        if (send_node >= num_nodes)
            send_node -= num_nodes;
        recv_node = rank_node - i;
        if (recv_node < 0)
            recv_node += num_nodes;

        send_pos = send_node * send_bytes_node;
        recv_pos = recv_node * recv_bytes_node;

        MPI_Sendrecv(send_buffer + send_pos, sendcount_node, sendtype, 
                send_node*PPN + local_rank, tag,
                tmpbuf + recv_pos, recvcount_node, recvtype,
                recv_node*PPN + local_rank, tag, 
                mpi_comm->global_comm, &status);
    }

    /************************************************
     * Step 2 : Redistribute received data within node
     ************************************************/
    for (int i = 0; i < num_nodes; i++)
        for (int j = 0; j < PPN; j++)
            memcpy(recv_buffer + ((j*num_nodes+i)*recv_bytes),
                    tmpbuf + ((i*PPN+j)*recv_bytes),
                    recv_bytes);

    for (int i = 0; i < PPN; i++)
    {
        send_proc = local_rank + i;
        if (send_proc >= PPN)
            send_proc -= PPN;
        recv_proc = local_rank - i;
        if (recv_proc < 0)
            recv_proc += PPN;

        send_pos = send_proc * recv_bytes * num_nodes;
        recv_pos = recv_proc * recv_bytes * num_nodes;

        MPI_Sendrecv(recv_buffer + send_pos, recvcount * num_nodes, recvtype,
                send_proc, tag,
                tmpbuf + recv_pos, recvcount * num_nodes, recvtype,
                recv_proc, tag,
                mpi_comm->local_comm, &status);

    }

    for (int i = 0; i < num_nodes; i++)
        for (int j = 0; j < PPN; j++)
            memcpy(recv_buffer + ((i*PPN+j)*recv_bytes),
                    tmpbuf + ((j*num_nodes+i)*recv_bytes),
                    recv_bytes);

    free(tmpbuf);

    return 0;
}

