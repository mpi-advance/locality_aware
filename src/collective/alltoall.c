#include "alltoall.h"
#include <string.h>
#include <math.h>

#ifdef GPU
#include "heterogeneous/gpu_alltoall.h"
#endif

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
int MPIX_Alltoall(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* mpi_comm)
{    
#ifdef GPU
    gpuMemoryType send_type, recv_type;
    get_mem_types(sendbuf, recvbuf, &send_type, &recv_type);

    if (send_type == gpuMemoryTypeDevice && 
            recv_type == gpuMemoryTypeDevice)
    {
        return copy_to_cpu_alltoall_pairwise(sendbuf,
                sendcount,
                sendtype,
                recvbuf,
                recvcount,
                recvtype,
                mpi_comm);
    }
    else if (send_type == gpuMemoryTypeDevice ||
            recv_type == gpuMemoryTypeDevice)
    {
        return gpu_aware_alltoall_pairwise(sendbuf,
                sendcount,
                sendtype,
                recvbuf,
                recvcount,
                recvtype,
                mpi_comm);
    }
#endif
    return alltoall_pairwise(sendbuf,
        sendcount,
        sendtype,
        recvbuf,
        recvcount,
        recvtype,
        mpi_comm);
}

int alltoall_pairwise(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int tag = 102944;
    int send_proc, recv_proc;
    int send_pos, recv_pos;
    MPI_Status status;

    char* recv_buffer = (char*)recvbuf;
    char* send_buffer = (char*)sendbuf;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

#ifdef GPU
    gpuMemoryType send_type, recv_type;
    gpuMemcpyKind memcpy_kind;
    get_mem_types(sendbuf, recvbuf, &send_type, &recv_type);
   
    if (send_type == gpuMemoryTypeDevice ||
            recv_type == gpuMemoryTypeDevice)
    {
        get_memcpy_kind(send_type, recv_type, &memcpy_kind);
        int ierr = gpuMemcpy(recv_buffer + (rank * recvcount * recv_size),
                send_buffer + (rank * sendcount * send_size),
                sendcount * send_size,
                memcpy_kind);
        gpu_check(ierr);
    }
    else
#endif
    memcpy(recv_buffer + (rank * recvcount * recv_size),
        send_buffer + (rank * sendcount * send_size),
        sendcount * send_size);


    // Send to rank + i
    // Recv from rank - i
    for (int i = 1; i < num_procs; i++)
    {
        send_proc = rank + i;
        if (send_proc >= num_procs)
            send_proc -= num_procs;
        recv_proc = rank - i;
        if (recv_proc < 0)
            recv_proc += num_procs;
        send_pos = send_proc * sendcount * send_size;
        recv_pos = recv_proc * recvcount * recv_size;

        MPI_Sendrecv(send_buffer + send_pos, 
                sendcount, 
                sendtype, 
                send_proc, 
                tag,
                recv_buffer + recv_pos, 
                recvcount, 
                recvtype, 
                recv_proc, 
                tag,
                comm->global_comm, 
                &status);
    }
    return MPI_SUCCESS;
}

int alltoall_nonblocking(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int tag = 102944;
    int send_proc, recv_proc;
    int send_pos, recv_pos;

    char* recv_buffer = (char*)recvbuf;
    char* send_buffer = (char*)sendbuf;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    MPI_Request* requests = (MPI_Request*)malloc(2*(num_procs-1)*sizeof(MPI_Request));

#ifdef GPU
    gpuMemoryType send_type, recv_type;
    gpuMemcpyKind memcpy_kind;
    get_mem_types(sendbuf, recvbuf, &send_type, &recv_type);

    if (send_type == gpuMemoryTypeDevice ||
            recv_type == gpuMemoryTypeDevice)
    {
        recv_proc = rank - stride;
        if (recv_proc < 0) recv_proc += num_procs;
        send_proc = rank + stride;
        if (send_proc >= num_procs) send_proc -= num_procs;

        group_size = stride * recvcount;
        
        ctr = 0;
        for (int i = group_size; i < total_count; i += (group_size*2))
        {
            for (int j = 0; j < group_size; j++)
            {
                for (int k = 0; k < recv_size; k++)
                {
                    contig_buf[ctr*recv_size+k] = recv_buffer[(i+j)*recv_size+k];
                }
                ctr++;
            }
        }

        size = ((int)(total_count / group_size) * group_size) / 2;

        MPI_Isend(contig_buf, size, recvtype, send_proc, tag, comm, &(requests[0]));
        MPI_Irecv(tmpbuf, size, recvtype, recv_proc, tag, comm, &(requests[1]));
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

        ctr = 0;
        for (int i = group_size; i < total_count; i += (group_size*2))
        {
            for (int j = 0; j < group_size; j++)
            {
                for (int k = 0; k < recv_size; k++)
                {
                    recv_buffer[(i+j)*recv_size+k] = tmpbuf[ctr*recv_size+k];
                }
                ctr++;
            }
        }

        stride *= 2;

    } 

    // 3. rotate local data
    if (rank < num_procs)
        rotate(recv_buffer, (rank+1)*msg_size, num_procs*msg_size);


    // TODO :: REVERSE!

    return 0;
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

        MPI_Sendrecv(sendbuf + send_pos, sendcount_node, sendtype, 
                send_node*PPN + local_rank, tag,
                tmpbuf + recv_pos, recvcount_node, recvtype,
                recv_node*PPN + local_rank, tag, 
                mpi_comm->global_comm, &status);
    }
    else
#endif
    memcpy(recv_buffer + (rank * recvcount * recv_size),
        send_buffer + (rank * sendcount * send_size),
        sendcount * send_size);

    // Send to rank + i
    // Recv from rank - i
    for (int i = 1; i < num_procs; i++)
    {
        send_proc = rank + i;
        if (send_proc >= num_procs)
            send_proc -= num_procs;
        recv_proc = rank - i;
        if (recv_proc < 0)
            recv_proc += num_procs;
        send_pos = send_proc * sendcount * send_size;
        recv_pos = recv_proc * recvcount * recv_size;

        MPI_Isend(send_buffer + send_pos,
                sendcount, 
                sendtype, 
                send_proc,
                tag, 
                comm->global_comm,
                &(requests[i-1]));
        MPI_Irecv(recv_buffer + recv_pos,
                recvcount,
                recvtype,
                recv_proc,
                tag,
                comm->global_comm,
                &(requests[num_procs + i - 2]));
    }

    MPI_Waitall(2*(num_procs-1), requests, MPI_STATUSES_IGNORE);

    free(requests);
    return 0;
}

