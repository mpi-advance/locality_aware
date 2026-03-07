#include "collective/allreduce_init.h"
#include "locality_aware.h"
#include <string.h>
#include <math.h>

int allreduce_recursive_doubling_init(const void* sendbuf,
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

    return allreduce_recursive_doubling_init_helper(sendbuf, recvbuf, count,
            datatype, op, comm, info, req_ptr, MPIL_Alloc, MPIL_Free);
        
}


int allreduce_recursive_doubling_init_helper(const void* sendbuf,
                                 void* recvbuf,
                                 int count,
                                 MPI_Datatype datatype,
                                 MPI_Op op,
                                 MPIL_Comm* comm,
                                 MPIL_Info* info,
                                 MPIL_Request** req_ptr,
                                 MPIL_Alloc_ftn alloc_ftn,
                                 MPIL_Free_ftn free_ftn)
{

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
    allocate_requests(1, local_S_request);
    allocate_requests(1, local_R_request);

    request->start_function = allreduce_recursive_doubling_start;
    request->wait_function  = allreduce_recursive_doubling_wait;

    request->count = count;
    request->op = op;
    request->datatype = datatype;
    request->sendbuf = sendbuf;
    request->recvbuf = recvbuf;


    int type_size;
    MPI_Type_size(datatype, &type_size);

    alloc_ftn(&(request->tmpbuf), type_size*count);
    request->free_ftn = free_ftn;

    int tag;
    get_tag(comm, &tag);

    request->n_msgs = 0;
    if (sendbuf != MPI_IN_PLACE)
    {
        MPI_Send_init(sendbuf, count, datatype, rank, tag,
                comm->global_comm, &(local_L_request->requests[local_L_request->n_msgs++]));
        MPI_Recv_init(recvbuf, count, datatype, rank, tag,
                comm->global_comm, &(local_L_request->requests[local_L_request->n_msgs++]));
    }

    int proc; 
    int log_procs = (int)log2(num_procs);
    int log2_num_procs = 1 << log_procs;
    int extra_procs = num_procs - log2_num_procs;

    if (rank >= log2_num_procs)
    {
        proc = rank - log2_num_procs;
        MPI_Send_init(recvbuf, count, datatype, rank - log2_num_procs, tag, 
                comm->global_comm, &(local_S_request->requests[local_S_request->n_msgs++]));
        MPI_Recv_init(recvbuf, count, datatype, rank - log2_num_procs, tag, 
                comm->global_comm, &(local_R_request->requests[local_R_request->n_msgs++]));
    }
    else
    {
        if (rank < extra_procs)
        {
            MPI_Recv_init(request->tmpbuf, count, datatype, rank + log2_num_procs, tag, 
                    comm->global_comm, &(local_S_request->requests[local_S_request->n_msgs++]));
        }
        for (int stride = 1; stride < log2_num_procs; stride = stride << 1)
        {
            proc = rank ^ stride;
            MPI_Send_init(recvbuf, count, datatype, proc, tag, comm->global_comm, 
                    &(request->requests[request->n_msgs++]));
            MPI_Recv_init(request->tmpbuf, count, datatype, proc, tag, comm->global_comm, 
                    &(request->requests[request->n_msgs++]));
        }
        if (rank < extra_procs)
        {
            MPI_Send_init(recvbuf, count, datatype, rank + log2_num_procs, tag, 
                    comm->global_comm, &(local_R_request->requests[local_R_request->n_msgs++]));
        }
    }

    *req_ptr = request;    

    return MPI_SUCCESS;
}

int allreduce_recursive_doubling_start(MPIL_Request* request)
{
    MPIL_Request* local_L_request = request->local_L_request;
    MPIL_Request* local_S_request = request->local_S_request;
    MPIL_Request* local_R_request = request->local_R_request;

    int type_size;
    MPI_Type_size(request->datatype, &type_size);

#if defined(GPU)
if (request->gpu_sendbuf)
{
#if defined(APU)
    memcpy(request->tmp_sendbuf, request->gpu_sendbuf, request->count*type_size);
#else
    gpuMemcpyAsync(request->tmp_sendbuf, request->gpu_sendbuf, request->count*type_size, 
            gpuMemcpyDeviceToHost, 0);
    gpuStreamSynchronize(0);
#endif
}
#endif
    if (request == NULL)
        return 0;

    if (local_L_request->n_msgs)
        MPI_Startall(local_L_request->n_msgs, local_L_request->requests);

    if (local_S_request->n_msgs)
        MPI_Startall(local_S_request->n_msgs, local_S_request->requests);

    return MPI_SUCCESS;
}

int allreduce_recursive_doubling_wait(MPIL_Request* request, MPI_Status* status)   
{
    MPIL_Request* local_L_request = request->local_L_request;
    MPIL_Request* local_S_request = request->local_S_request;
    MPIL_Request* local_R_request = request->local_R_request;

    int type_size;
    MPI_Type_size(request->datatype, &type_size);

    if (request == NULL)
        return 0;

    if (local_L_request->n_msgs)
        MPI_Waitall(local_L_request->n_msgs, local_L_request->requests, MPI_STATUSES_IGNORE);

    if (local_S_request->n_msgs)
    {
        MPI_Waitall(local_S_request->n_msgs, local_S_request->requests, MPI_STATUSES_IGNORE);
        MPI_Reduce_local(request->tmpbuf, request->recvbuf, request->count,
                request->datatype, request->op);
    }

    for (int i = 0; i < request->n_msgs; i += 2)
    {
        MPI_Startall(2, &(request->requests[i]));
        MPI_Waitall(2, &(request->requests[i]), MPI_STATUSES_IGNORE);
        MPI_Reduce_local(request->tmpbuf, request->recvbuf, request->count,
                request->datatype, request->op);
    }
    

    if (local_R_request->n_msgs)
    {
        MPI_Startall(local_R_request->n_msgs, local_R_request->requests);
        MPI_Waitall(local_R_request->n_msgs, local_R_request->requests, MPI_STATUSES_IGNORE);
    }

#if defined(GPU)
if (request->gpu_recvbuf)
{
#if defined(APU)
    memcpy(request->gpu_recvbuf, request->recvbuf, request->count*type_size);
#else
    gpuMemcpyAsync(request->gpu_recvbuf, request->recvbuf, request->count*type_size, 
            gpuMemcpyHostToDevice, 0);
    gpuStreamSynchronize(0);
#endif
}
#endif

    return MPI_SUCCESS;
}

