#include "collective/alltoallv_init.h"
#include "locality_aware.h"

// NOTE: No GPU-Aware Version, but Copy To CPU would work
int alltoallv_init_rma(const void* sendbuf,
                       const int sendcounts[],
                       const int sdispls[],
                       MPI_Datatype sendtype,
                       void* recvbuf,
                       const int recvcounts[],
                       const int rdispls[],
                       MPI_Datatype recvtype,
                       MPIL_Comm* comm,
                       MPIL_Info* info,
                       MPIL_Request** req_ptr)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    MPIL_Request* request;
    init_request(&request);
    
    request->start_function = alltoallv_rma_start;
    request->wait_function = alltoallv_rma_wait;

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    request->sendbuf = sendbuf;
    request->recvbuf = recvbuf;
    request->n_puts = num_procs;
    request->sdispls = (int*)malloc(num_procs*sizeof(int));
    request->put_displs = (int*)malloc(num_procs*sizeof(int));
    request->put_bytes = (int*)malloc(num_procs*sizeof(int));

    int bytes = 0;
    for (int i = 0; i < num_procs; i++)
        bytes += (recvcounts[i] * recv_bytes);
    MPIL_Request_win_init(request, recvbuf, bytes, 1, comm->global_comm);

    for (int i = 0; i < num_procs; i++)
    {
        request->sdispls[i] = sdispls[i] * send_bytes;
        request->put_bytes[i] = sendcounts[i] * send_bytes;
    }
    MPIL_Alltoall(rdispls, 1, MPI_INT, request->put_displs, 1, MPI_INT, comm);
    for (int i = 0; i < num_procs; i++)
    {
        request->put_displs[i] *= recv_bytes;
    }

    *req_ptr = request;

    return MPI_SUCCESS;

}


int alltoallv_rma_start(MPIL_Request* request)
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

    for (int i = 0; i < request->n_puts; ++i) {
        MPI_Put(send_buffer + request->sdispls[i],
                request->put_bytes[i],
                MPI_BYTE,
                i,
                request->put_displs[i],
                request->put_bytes[i],
                MPI_BYTE,
                request->win);
    }

    return MPI_SUCCESS;
}


int alltoallv_rma_wait(MPIL_Request* request, MPI_Status* status)
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
    
