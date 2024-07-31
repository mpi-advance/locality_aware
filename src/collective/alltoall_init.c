#include "alltoall_init.h"
#include <string.h>
#include <math.h>

#ifdef GPU
#include "heterogeneous/gpu_alltoall_init.h"
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
int MPIX_Alltoall_init(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* mpi_comm,
        MPI_Info info,
        MPIX_Request** request_ptr)
{   
/*
#ifdef GPU
    gpuMemoryType send_type, recv_type;
    get_mem_types(sendbuf, recvbuf, &send_type, &recv_type);

    if (send_type == gpuMemoryTypeDevice && 
            recv_type == gpuMemoryTypeDevice)
    {
        return copy_to_cpu_alltoall_init_pairwise(sendbuf,
                sendcount,
                sendtype,
                recvbuf,
                recvcount,
                recvtype,
                mpi_comm,
                info,
                request_ptr);
    }
    else if (send_type == gpuMemoryTypeDevice ||
            recv_type == gpuMemoryTypeDevice)
    {
        return gpu_aware_alltoall_init_stride(sendbuf,
                sendcount,
                sendtype,
                recvbuf,
                recvcount,
                recvtype,
                mpi_comm,
                info,
                request_ptr);
    }
#endif
*/
    return alltoall_init_stride(sendbuf,
        sendcount,
        sendtype,
        recvbuf,
        recvcount,
        recvtype,
        mpi_comm->global_comm, 
        info,
        request_ptr);
}

int alltoall_init_nonblocking(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm,
        MPI_Info info,
        MPIX_Request** request_ptr)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    init_request(request_ptr);
    MPIX_Request* request = *request_ptr;
    request->global_n_msgs = 2*num_procs;
    allocate_requests(request->global_n_msgs, &(request->global_requests));
    request->start_function = (void*) neighbor_start;
    request->wait_function = (void*) neighbor_wait;

    return alltoall_init_nonblocking_helper(sendbuf,
            sendcount,
            sendtype,
            recvbuf,
            recvcount,
            recvtype,
            comm,
            info,
            request_ptr);
}

int alltoall_init_stride(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm,
        MPI_Info info,
        MPIX_Request** request_ptr)
{   
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);
    
    init_request(request_ptr);
    MPIX_Request* request = *request_ptr;
    request->global_n_msgs = 2*num_procs;
    allocate_requests(request->global_n_msgs, &(request->global_requests));
    request->start_function = (void*) partial_neighbor_start;
    request->wait_function = (void*) partial_neighbor_wait;

    request->block_size = 1;
            
    return alltoall_init_nonblocking_helper(sendbuf,
            sendcount,
            sendtype,
            recvbuf,
            recvcount,
            recvtype,
            comm,
            info,
            request_ptr);
}

int alltoall_init_nonblocking_helper(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm,
        MPI_Info info,
        MPIX_Request** request_ptr)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    MPIX_Request* request = *request_ptr;

    int tag = 102944;
    int send_proc, recv_proc;
    int send_pos, recv_pos;

    char* recv_buffer = (char*)recvbuf;
    char* send_buffer = (char*)sendbuf;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

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

        MPI_Send_init(send_buffer + send_pos,
                sendcount, 
                sendtype, 
                send_proc,
                tag, 
                comm,
                &(request->global_requests[i]));
        MPI_Recv_init(recv_buffer + recv_pos,
                recvcount,
                recvtype,
                recv_proc,
                tag,
                comm,
                &(request->global_requests[num_procs+i]));
    }

    return MPI_SUCCESS;
}

int partial_neighbor_start(MPIX_Request* request)
{
    int block_size = request->block_size;
    int remaining = request->global_n_msgs;
    int current = 0;

    while (remaining)
    {
        if (remaining < block_size)
            block_size = remaining;

        MPI_Startall(block_size, &(request->global_requests[current]));
        MPI_Waitall(block_size, &(request->global_requests[current]), MPI_STATUSES_IGNORE);

        current += block_size;
        remaining -= block_size;
    }
 
    return MPI_SUCCESS;
}

int partial_neighbor_wait(MPIX_Request* request, MPI_Status status)
{
    return MPI_SUCCESS;
}
