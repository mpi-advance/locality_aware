#include "collective/alltoall_init.h"
#include "heterogeneous/gpu_utils.h"
#include "locality_aware.h"

int alltoall_init(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIL_Comm* xcomm,
        MPIL_Info* xinfo,
        MPIL_Request** request_ptr)
{
    int rank, num_procs;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_size(xcomm->global_comm, &num_procs);

    MPIL_Request* request;
    init_request(&request);
    allocate_requests(2*num_procs, request);
    request->n_msgs = 2*num_procs;
    request->sendbuf = sendbuf;
    request->recvbuf = recvbuf;

    int tag;
    MPIL_Comm_tag(xcomm, &tag);

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
                xcomm->global_comm, &(request->requests[2*i]));
        MPI_Recv_init(recv_buffer + recv_pos, recvcount, recvtype, recv_proc, tag,
                xcomm->global_comm, &(request->requests[2*i + 1]));
    }

    *request_ptr = request;

    return MPI_SUCCESS;
}

int alltoall_init_pairwise(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIL_Comm* xcomm,
        MPIL_Info* xinfo,
        MPIL_Request** request_ptr)
{
    
    alltoall_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, xcomm,
            xinfo, request_ptr);
    MPIL_Request* request = *request_ptr;
    request->start_function = alltoall_pairwise_start;
    request->wait_function = alltoall_pairwise_wait;

    return MPI_SUCCESS;
}

int alltoall_init_nonblocking(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIL_Comm* xcomm,
        MPIL_Info* xinfo,
        MPIL_Request** request_ptr)
{
    int num_procs;
    MPI_Comm_size(xcomm->global_comm, &num_procs);

    alltoall_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, xcomm,
            xinfo, request_ptr);
    MPIL_Request* request = *request_ptr;
    request->start_function = alltoall_nonblocking_start;
    request->wait_function = alltoall_nonblocking_wait;


    return MPI_SUCCESS;
}

int alltoall_nonblocking_start(MPIL_Request* request)
{
    if (request == NULL)
    {
        return MPI_SUCCESS;
    }

#if defined(GPU)
if (request->gpu_sendbuf)
{
    int gpu_error;
#if defined(APU)
    memcpy(request->tmp_gpubuf, request->gpu_sendbuf, request->size_sends);
#else
    gpu_error = gpuMemcpyAsync(request->tmp_gpubuf, request->gpu_sendbuf, request->size_sends, 
            gpuMemcpyDeviceToHost, 0);
    gpu_check(gpu_error);
    gpu_error = gpuStreamSynchronize(0);
    gpu_check(gpu_error);
#endif
}
#endif

    if (request->n_msgs)
        MPI_Startall(request->n_msgs, request->requests);
    return MPI_SUCCESS;
}

int alltoall_nonblocking_wait(MPIL_Request* request, MPI_Status* status)
{
    if (request == NULL)
    {
        return MPI_SUCCESS;
    }

    if (request->n_msgs)
        MPI_Waitall(request->n_msgs, request->requests, MPI_STATUSES_IGNORE);

#if defined(GPU)
    int gpu_error;
if (request->gpu_recvbuf)
{
#if defined(APU)
    memcpy(request->gpu_recvbuf, request->recvbuf, request->size_recvs);
#else
    gpu_error = gpuMemcpyAsync(request->gpu_recvbuf, request->recvbuf, request->size_recvs, 
            gpuMemcpyHostToDevice, 0);
    gpu_check(gpu_error);
    gpu_error = gpuStreamSynchronize(0);
    gpu_check(gpu_error);
#endif
}
#endif

    return MPI_SUCCESS;
}

int alltoall_pairwise_start(MPIL_Request* request)
{
#if defined(GPU)
int gpu_error;
if (request->gpu_sendbuf)
{
#if defined(APU)
    memcpy(request->tmp_gpubuf, request->gpu_sendbuf, request->size_sends);
#else
    gpu_error = gpuMemcpyAsync(request->tmp_gpubuf, request->gpu_sendbuf, request->size_sends, 
            gpuMemcpyDeviceToHost, 0);
    gpu_check(gpu_error);
    gpu_error = gpuStreamSynchronize(0);
    gpu_check(gpu_error);
#endif
}
#endif
    if (request->n_msgs)
        MPI_Startall(2, request->requests);
    return MPI_SUCCESS;
}

int alltoall_pairwise_wait(MPIL_Request* request, MPI_Status* status)
{
    if (request->n_msgs)
        MPI_Waitall(2, request->requests, MPI_STATUSES_IGNORE);

    for (int i = 2; i < request->n_msgs; i += 2)
    {
        MPI_Startall(2, &(request->requests[i]));
        MPI_Waitall(2, &(request->requests[i]), MPI_STATUSES_IGNORE);
    }
#if defined(GPU)
int gpu_error;
if (request->gpu_recvbuf)
{
#if defined(APU)
    memcpy(request->gpu_recvbuf, request->recvbuf, request->size_recvs);
#else
    gpu_error = gpuMemcpyAsync(request->gpu_recvbuf, request->recvbuf, request->size_recvs, 
            gpuMemcpyHostToDevice, 0);
    gpu_check(gpu_error);
    gpu_error = gpuStreamSynchronize(0);
    gpu_check(gpu_error);
#endif
}
#endif

    return MPI_SUCCESS;
}
