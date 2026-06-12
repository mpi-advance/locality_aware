#include "collective/allreduce_init.h"
#include "heterogeneous/gpu_utils.h"
#include "locality_aware.h"
#include <string.h>
#include <math.h>

int allreduce_init_dissemination_radix(const void* sendbuf,
                                 void* recvbuf,
                                 int count,
                                 MPI_Datatype datatype,
                                 MPI_Op op,
                                 MPIL_Comm* comm,
                                 MPIL_Info* info,
                                 MPIL_Request** req_ptr)
{

    if (count == 0)
        return MPI_SUCCESS;

    int radix = mpil_collective_radix;

    int tag;
    get_tag(comm, &tag);

    int type_size;
    MPI_Type_size(datatype, &type_size);

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


    int max_outer_steps = (int)(log((double)num_procs) / log((double)radix)) + 1;
    int max_global_msgs = max_outer_steps * 2 * (radix - 1);
    allocate_requests(max_global_msgs, request);
    allocate_requests(2, local_L_request);
    allocate_requests(1, local_S_request);
    allocate_requests(2, local_R_request);
    request->n_msgs = 0;
    local_L_request->n_msgs = 0;
    local_S_request->n_msgs = 0;
    local_R_request->n_msgs = 0;

    request->start_function = allreduce_dissemination_radix_start;
    request->wait_function = allreduce_dissemination_radix_wait;

    request->count = count;
    request->op = op;
    request->datatype = datatype;
    request->sendbuf = sendbuf;
    request->recvbuf = recvbuf;
    MPI_Comm_dup(comm->global_comm, &(request->global_comm));
    MPIL_Alloc(&(request->tmpbuf), radix*type_size*count*num_procs);


    if (sendbuf != MPI_IN_PLACE)
    {
        MPI_Send_init(sendbuf, count, datatype, rank, tag, 
                comm->global_comm, &(local_L_request->requests[local_L_request->n_msgs++]));
        MPI_Recv_init(recvbuf, count, datatype, rank, tag,
                comm->global_comm, &(local_L_request->requests[local_L_request->n_msgs++]));
    }

    int pow_radix_num_procs = 1;
    while (pow_radix_num_procs * radix <= num_procs)
        pow_radix_num_procs *= radix;
    int mult = num_procs / pow_radix_num_procs;
    int max_proc = mult * pow_radix_num_procs;
    int extra = num_procs - max_proc;

    request->num_ops = radix;
    request->recv_size = max_proc;

    if (rank >= max_proc)
    {
        int proc = rank - max_proc;
        MPI_Send_init(recvbuf, count, datatype, proc, tag, 
            comm->global_comm, &(local_S_request->requests[local_S_request->n_msgs++]));
        MPI_Recv_init(recvbuf, count, datatype, proc, tag,
            comm->global_comm, &(local_R_request->requests[local_R_request->n_msgs++]));
    }
    else
    {
        if (rank < extra)
        {
            MPI_Recv_init(request->tmpbuf, count, datatype,  max_proc + rank, tag,
                    comm->global_comm, &(local_S_request->requests[local_S_request->n_msgs++]));
        }

        for (int stride_start = 1; stride_start < max_proc; stride_start *= radix)
        {
            for (int step = 1; step < radix; step++)
            {
                int stride = stride_start * step;
                if (stride < max_proc)
                {
                    int send_proc = (rank - stride + max_proc) % max_proc;
                    int recv_proc = (rank + stride) % max_proc;
                    MPI_Send_init(recvbuf, count, datatype, send_proc, tag,
                            comm->global_comm,
                            &(request->requests[request->n_msgs++]));
                    MPI_Recv_init((char*)request->tmpbuf + (step-1) * count * type_size,
                            count, datatype, recv_proc, tag,
                            comm->global_comm,
                            &(request->requests[request->n_msgs++]));
                }
            }
        }

        if (rank < extra)
        {
            MPI_Send_init(recvbuf, count, datatype, max_proc + rank, tag, 
                    comm->global_comm, &(local_R_request->requests[local_R_request->n_msgs++]));
        }
    }

    *req_ptr = request;

    return MPI_SUCCESS;
}


int allreduce_dissemination_radix_start(MPIL_Request* request)
{
    MPIL_Request* local_L_request = request->local_L_request;
    MPIL_Request* local_S_request = request->local_S_request;
    MPIL_Request* local_R_request = request->local_R_request;

    int type_size;
    MPI_Type_size(request->datatype, &type_size);

#if defined(GPU)
if (request->gpu_sendbuf)
{
    int gpu_error;
#if defined(APU)
    memcpy(request->tmp_gpubuf, request->gpu_sendbuf, 
            request->size_sends);
#else
    gpu_error = gpuMemcpyAsync(request->tmp_gpubuf, request->gpu_sendbuf, 
            request->size_sends, gpuMemcpyDeviceToHost, 0);
    gpu_check(gpu_error);
    gpu_error = gpuStreamSynchronize(0);
    gpu_check(gpu_error);
#endif
}
#endif

    if (request == NULL)
        return MPI_SUCCESS;

    if (local_L_request->n_msgs)
        MPI_Startall(local_L_request->n_msgs, local_L_request->requests);

    if (local_S_request->n_msgs)
        MPI_Startall(local_S_request->n_msgs, local_S_request->requests);

    return MPI_SUCCESS;
}

int allreduce_dissemination_radix_wait(MPIL_Request* request, MPI_Status* status)
{
    MPIL_Request* local_L_request = request->local_L_request;
    MPIL_Request* local_S_request = request->local_S_request;
    MPIL_Request* local_R_request = request->local_R_request;

    if (request == NULL)
        return MPI_SUCCESS;

    int type_size;
    MPI_Type_size(request->datatype, &type_size);

    int radix = request->num_ops;
    int max_proc = request->recv_size;

    if (local_L_request->n_msgs)
        MPI_Waitall(local_L_request->n_msgs, local_L_request->requests, MPI_STATUSES_IGNORE);

    if (local_S_request->n_msgs)
    {
        MPI_Waitall(local_S_request->n_msgs, local_S_request->requests, MPI_STATUSES_IGNORE);  
        MPI_Reduce_local(request->tmpbuf, request->recvbuf, request->count,
                request->datatype, request->op);
    }

    int ctr = 0;
    for (int stride_start = 1; stride_start < max_proc; stride_start *= radix)
    {
        int n_msgs = 0;
        for (int step = 1; step < radix; step++)
        {
            int stride = stride_start * step;
            if (stride < max_proc)
                n_msgs += 2;
        }

        MPI_Startall(n_msgs, &(request->requests[ctr]));
        MPI_Waitall(n_msgs, &(request->requests[ctr]),
                MPI_STATUSES_IGNORE);

        for (int i = 0; i < n_msgs / 2; i++)
        {
            MPI_Reduce_local((char*)request->tmpbuf + i * request->count * type_size,
                    request->recvbuf, request->count, request->datatype, request->op);

        }
        ctr += n_msgs;
    }

    if (local_R_request->n_msgs)
    {
        MPI_Startall(local_R_request->n_msgs, local_R_request->requests);
        MPI_Waitall(local_R_request->n_msgs, local_R_request->requests, MPI_STATUSES_IGNORE);
    }

#if defined(GPU)
    int gpu_error;
if (request->gpu_recvbuf)
{
#if defined(APU)
    memcpy(request->gpu_recvbuf, request->recvbuf, 
            request->size_recvs);
#else
    gpu_error = gpuMemcpyAsync(request->gpu_recvbuf, request->recvbuf, 
            request->size_recvs, gpuMemcpyHostToDevice, 0);
    gpu_check(gpu_error);
    gpu_error = gpuStreamSynchronize(0);
    gpu_check(gpu_error);
#endif
}
#endif

    return MPI_SUCCESS;
}

