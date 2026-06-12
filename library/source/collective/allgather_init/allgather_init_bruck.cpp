#include "collective/allgather_init.h"
#include "heterogeneous/gpu_utils.h"
#include "locality_aware.h"
#include <math.h>
// Implements the Bruck allgather algorithm
// Note: current implementation will not work for non-contiguous datatypes
// To work with non-contig data, need to get extent for recvbuf to index into it
int allgather_init_bruck(const void* sendbuf,
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

    MPIL_Request* request;
    init_request(&request);
    init_request(&(request->local_L_request));
    init_request(&(request->local_S_request));
    init_request(&(request->local_R_request));
    MPIL_Request* local_L_request = request->local_L_request;
    MPIL_Request* local_S_request = request->local_S_request;
    MPIL_Request* local_R_request = request->local_R_request;

    int max_n_msgs = 2*(log2(num_procs));
    allocate_requests(max_n_msgs, request);
    allocate_requests(2, local_L_request);
    allocate_requests(2, local_S_request);
    allocate_requests(2, local_R_request);
    request->n_msgs = 0;
    local_L_request->n_msgs = 0;
    local_S_request->n_msgs = 0;
    local_R_request->n_msgs = 0;

    request->start_function = allgather_bruck_start;
    request->wait_function  = allgather_bruck_wait;

    request->recvbuf = recvbuf;

    int tag;
    get_tag(comm, &tag);

    int bytes;
    MPI_Type_size(recvtype, &bytes);
    int count_bytes = recvcount * bytes;

    char* _recvbuf = (char*)recvbuf;
    MPIL_Alloc((void**)&(request->tmpbuf), count_bytes*num_procs);

    // Sendrecv instead of memcpy, so that it works on the GPUs
    if (sendbuf != MPI_IN_PLACE)
    {
        MPI_Send_init(sendbuf, sendcount, sendtype, rank, tag,
                comm->global_comm, &(local_L_request->requests[local_L_request->n_msgs++]));
        MPI_Recv_init(request->tmpbuf, recvcount, recvtype, rank, tag,
                comm->global_comm, &(local_L_request->requests[local_L_request->n_msgs++]));
    }

    int log_procs = (int)log2(num_procs);
    int log2_num_procs = 1 << log_procs;    

    int pow_i = 1;
    int send_proc, recv_proc;
    for (int i = 0; i < log_procs; i++)
    {
        send_proc = (rank - pow_i + num_procs) % num_procs;
        recv_proc = (rank + pow_i) % num_procs;

        MPI_Send_init(request->tmpbuf, recvcount * pow_i, recvtype, send_proc, tag,
                comm->global_comm, &(request->requests[request->n_msgs++]));
        MPI_Recv_init(((char*)request->tmpbuf) + pow_i * count_bytes, recvcount * pow_i, recvtype, recv_proc, tag,
                comm->global_comm, &(request->requests[request->n_msgs++]));

        pow_i *= 2;
    }

    // If non-power-of-2 process count, 1 more step
    if (log2_num_procs != num_procs)
    {
        int count = num_procs - log2_num_procs;
        send_proc = (rank - pow_i + num_procs) % num_procs;
        recv_proc = (rank + pow_i) % num_procs;

        MPI_Send_init(request->tmpbuf, recvcount * count, recvtype, send_proc, tag,
                comm->global_comm, &(request->requests[request->n_msgs++]));
        MPI_Recv_init(((char*)request->tmpbuf) + pow_i * count_bytes, recvcount * count, recvtype, recv_proc, tag,
                comm->global_comm, &(request->requests[request->n_msgs++]));

    }

    int n_first_group = num_procs - rank;
    int n_last_group = num_procs - n_first_group;

    // Sendrecvs instead of memcpys, so that it works on the GPUs    
    MPI_Send_init(request->tmpbuf, recvcount * n_first_group, recvtype, rank, tag,
            comm->global_comm, &(local_S_request->requests[local_S_request->n_msgs++]));
    MPI_Recv_init(_recvbuf + rank * count_bytes, recvcount * n_first_group, recvtype, rank, tag,
            comm->global_comm, &(local_S_request->requests[local_S_request->n_msgs++]));
    if (rank != 0)
    {
        MPI_Send_init(((char*)(request->tmpbuf)) + (n_first_group * count_bytes), recvcount * n_last_group, recvtype, rank, tag,
                comm->global_comm, &(local_R_request->requests[local_R_request->n_msgs++]));
        MPI_Recv_init(_recvbuf, recvcount * n_last_group, recvtype, rank, tag,
                comm->global_comm, &(local_R_request->requests[local_R_request->n_msgs++]));
    }
    
    *req_ptr = request;

    return MPI_SUCCESS;
}

int allgather_bruck_start(MPIL_Request* request)
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

int allgather_bruck_wait(MPIL_Request* request, MPI_Status* status)
{
    if (request->local_L_request->n_msgs)
        MPI_Waitall(request->local_L_request->n_msgs, request->local_L_request->requests,
                MPI_STATUSES_IGNORE);

    for (int i = 0; i < request->n_msgs; i += 2)
    {
        MPI_Startall(2, &(request->requests[i]));
        MPI_Waitall(2, &(request->requests[i]), MPI_STATUSES_IGNORE);
    }

    if (request->local_S_request->n_msgs)
    {
        MPI_Startall(request->local_S_request->n_msgs,
                request->local_S_request->requests);
        MPI_Waitall(request->local_S_request->n_msgs,
                request->local_S_request->requests,
                MPI_STATUSES_IGNORE);
    }
    if (request->local_R_request->n_msgs)
    {
        MPI_Startall(request->local_R_request->n_msgs,
                request->local_R_request->requests);
        MPI_Waitall(request->local_R_request->n_msgs,
                request->local_R_request->requests,
                MPI_STATUSES_IGNORE);
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
