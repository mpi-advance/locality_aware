#include "collective/alltoall_init.h"
#include "locality_aware.h"

// NOTE: No GPU-Aware Version, but Copy To CPU would work
int alltoall_init_rma(const void* sendbuf,
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
    
    request->start_function = alltoall_rma_start;
    request->wait_function = alltoall_rma_wait;

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    request->sendbuf = sendbuf;
    request->recvbuf = recvbuf;
    request->n_puts = num_procs;
    request->send_size = send_bytes * sendcount;
    request->count = rank;

    int bytes = num_procs * recvcount * recv_bytes;
    MPIL_Request_win_init(request, recvbuf, bytes, 1, xcomm->global_comm);

    *request_ptr = request;

    return MPI_SUCCESS;

}


int alltoall_rma_start(MPIL_Request* request)
{
#if defined(GPU)
if (request->gpu_sendbuf)
{
#if defined(APU)
    memcpy(request->tmp_gpubuf, request->gpu_sendbuf, 
            request->n_puts * request->send_size);
#else
    int gpu_error;
    gpu_error = gpuMemcpyAsync(request->tmp_gpubuf, request->gpu_sendbuf, 
            request->n_puts * request->send_size, gpuMemcpyDeviceToHost, 0);
    gpu_check(gpu_error);
    gpu_error = gpuStreamSynchronize(0);
    gpu_check(gpu_error);
#endif
}
#endif
    
    const char*  send_buffer = (const char* )(request->sendbuf);

    MPI_Win_fence(MPI_MODE_NOPRECEDE, request->win);
    int nbytes = request->send_size;
    if (nbytes == 0) 
        return MPI_SUCCESS;

    for (int i = 0; i < request->n_puts; ++i) {
        MPI_Put(send_buffer + (nbytes*i),
                nbytes,
                MPI_BYTE,
                i,
                request->count * nbytes,
                nbytes,
                MPI_BYTE,
                request->win);
    }

    return MPI_SUCCESS;
}


int alltoall_rma_wait(MPIL_Request* request, MPI_Status* status)
{
    MPI_Win_fence(MPI_MODE_NOSUCCEED, request->win);

#if defined(GPU)
if (request->gpu_recvbuf)
{
#if defined(APU)
    memcpy(request->gpu_recvbuf, request->win_array, request->win_bytes);
#else
    int gpu_error;
    gpu_error = gpuMemcpyAsync(request->gpu_recvbuf, request->win_array, 
            request->win_bytes, gpuMemcpyHostToDevice, 0);
    gpu_check(gpu_error);
    gpu_error = gpuStreamSynchronize(0);
    gpu_check(gpu_error);
#endif
}
#endif


    return MPI_SUCCESS;
}
