#include "collective/allgather_init.h"
#include "heterogeneous/gpu_utils.h"
#include "locality_aware.h"

// Calls underlying MPI implementation
int allgather_init_ring(const void* sendbuf,
                   int sendcount,
                   MPI_Datatype sendtype,
                   void* recvbuf,
                   int recvcount,
                   MPI_Datatype recvtype,
                   MPIL_Comm* comm,
                   MPIL_Info* info,
                   MPIL_Request** req_ptr)
{
    if (sendcount == 0)
        return MPI_SUCCESS;

    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int send_proc = (rank + num_procs - 1) % num_procs;
    int recv_proc = (rank + 1) % num_procs;

    int tag;
    get_tag(comm, &tag);

    int bytes;
    MPI_Type_size(recvtype, &bytes);
    int count_bytes = recvcount * bytes;

    char* _recvbuf = (char*)recvbuf;

    MPIL_Request* request;
    init_request(&request);
    init_request(&(request->local_L_request));
    MPIL_Request* local_L_request = request->local_L_request;
    allocate_requests(2*num_procs, request);
    allocate_requests(2, local_L_request);
    request->n_msgs = 0;
    local_L_request->n_msgs = 0;

    request->start_function = allgather_ring_start;
    request->wait_function = allgather_ring_wait;

    request->recvbuf = recvbuf;    

    // Send sendbuf to myself, instead of memcpy, to work on GPU
    if (sendbuf != MPI_IN_PLACE)
    {
        MPI_Send_init(sendbuf, sendcount, sendtype, rank, tag,
                comm->global_comm, &(local_L_request->requests[local_L_request->n_msgs++]));
        MPI_Recv_init(_recvbuf + (rank * count_bytes), recvcount, recvtype, rank, tag,
                comm->global_comm, &(local_L_request->requests[local_L_request->n_msgs++]));
    }

    int pos = rank;
    int next_pos = (rank + 1) % num_procs;

    for (int i = 1; i < num_procs; i++)
    {
        MPI_Send_init(_recvbuf + (pos * count_bytes), recvcount, recvtype, send_proc, tag,
                comm->global_comm, &(request->requests[request->n_msgs++]));
        MPI_Recv_init(_recvbuf + (next_pos * count_bytes), recvcount, recvtype, recv_proc, tag,
                comm->global_comm, &(request->requests[request->n_msgs++]));
        pos = next_pos;
        next_pos = (next_pos + 1) % num_procs;
    }

    *req_ptr = request;

    return MPI_SUCCESS;
}

int allgather_ring_start(MPIL_Request* request)
{
#if defined(GPU)
if (request->gpu_sendbuf)
{
#if defined(APU)
    memcpy(request->tmp_gpubuf, request->gpu_sendbuf, request->size_sends);
#else
    gpuMemcpyAsync(request->tmp_gpubuf, request->gpu_sendbuf, request->size_sends, 
            gpuMemcpyDeviceToHost, 0);
    gpuStreamSynchronize(0);
#endif
}
#endif
    if (request->local_L_request->n_msgs)
        MPI_Startall(request->local_L_request->n_msgs, request->local_L_request->requests);
    return MPI_SUCCESS;
}

int allgather_ring_wait(MPIL_Request* request, MPI_Status* status)
{
    if (request->local_L_request->n_msgs)
        MPI_Waitall(request->local_L_request->n_msgs, request->local_L_request->requests,
                MPI_STATUSES_IGNORE);

    for (int i = 0; i < request->n_msgs; i += 2)
    {
        MPI_Startall(2, &(request->requests[i]));
        MPI_Waitall(2, &(request->requests[i]), MPI_STATUSES_IGNORE);
    }
#if defined(GPU)
if (request->gpu_recvbuf)
{
#if defined(APU)
    memcpy(request->gpu_recvbuf, request->recvbuf, request->size_recvs);
#else
    gpuMemcpyAsync(request->gpu_recvbuf, request->recvbuf, request->size_recvs, 
            gpuMemcpyHostToDevice, 0);
    gpuStreamSynchronize(0);
#endif
}
#endif

    return MPI_SUCCESS;
}
