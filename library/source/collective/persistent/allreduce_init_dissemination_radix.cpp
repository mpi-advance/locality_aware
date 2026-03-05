#include "collective/allreduce_init.h"
#include "locality_aware.h"
#include <string.h>
#include <math.h>

int allreduce_dissemination_radix_init(const void* sendbuf,
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

    return allreduce_dissemination_radix_init_helper(sendbuf, recvbuf,
            count, datatype, op, comm, info, req_ptr, MPIL_Alloc, MPIL_Free);
}

int allreduce_dissemination_radix_init_helper(const void* sendbuf,
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

    int max_outer_steps = (int)(log((double)num_procs) / log((double)radix)) + 1;
    int max_global_msgs = max_outer_steps * 2 * (radix - 1);
    allocate_requests(max_global_msgs, &(request->global_requests));
    allocate_requests(2, &(request->local_L_requests));
    allocate_requests(1, &(request->local_S_requests));
    allocate_requests(2, &(request->local_R_requests));

    request->start_function = allreduce_dissemination_radix_start;
    request->wait_function = allreduce_dissemination_radix_wait;

    request->count = count;
    request->op = op;
    request->datatype = datatype;
    request->sendbuf = sendbuf;
    request->global_comm = comm->global_comm;
    alloc_ftn(&(request->tmpbuf), radix*type_size*count*num_procs);
    request->free_ftn = free_ftn;


    if (sendbuf != MPI_IN_PLACE)
    {
        MPI_Send_init(sendbuf, count, datatype, rank, tag, 
                comm->global_comm, &(request->local_L_requests[request->local_L_n_msgs++]));
        MPI_Recv_init(recvbuf, count, datatype, rank, tag,
                comm->global_comm, &(request->local_L_requests[request->local_L_n_msgs++]));
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
            comm->global_comm, &(request->local_S_requests[request->local_S_n_msgs++]));
        MPI_Recv_init(recvbuf, count, datatype, proc, tag,
            comm->global_comm, &(request->local_R_requests[request->local_R_n_msgs++]));
    }
    else
    {
        if (rank < extra)
        {
            MPI_Recv_init(request->tmpbuf, count, datatype,  max_proc + rank, tag,
                    comm->global_comm, &(request->local_S_requests[request->local_S_n_msgs++]));
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
                            &(request->global_requests[request->global_n_msgs++]));
                    MPI_Recv_init((char*)request->tmpbuf + (step-1) * count * type_size,
                            count, datatype, recv_proc, tag,
                            comm->global_comm,
                            &(request->global_requests[request->global_n_msgs++]));
                }
            }
        }

        if (rank < extra)
        {
            MPI_Send_init(recvbuf, count, datatype, max_proc + rank, tag, 
                    comm->global_comm, &(request->local_R_requests[request->local_R_n_msgs++]));
        }
    }

    *req_ptr = request;

    return MPI_SUCCESS;
}


int allreduce_dissemination_radix_start(MPIL_Request* request)
{
    if (request == NULL)
        return MPI_SUCCESS;

    if (request->local_L_n_msgs)
        MPI_Startall(request->local_L_n_msgs, request->local_L_requests);

    if (request->local_S_n_msgs)
        MPI_Startall(request->local_S_n_msgs, request->local_S_requests);

    return MPI_SUCCESS;
}

int allreduce_dissemination_radix_wait(MPIL_Request* request, MPI_Status status)
{
    if (request == NULL)
        return MPI_SUCCESS;

    int type_size;
    MPI_Type_size(request->datatype, &type_size);

    int radix = request->num_ops;
    int max_proc = request->recv_size;

    if (request->local_L_n_msgs)
        MPI_Waitall(request->local_L_n_msgs, request->local_L_requests,
                MPI_STATUSES_IGNORE);

    if (request->local_S_n_msgs)
    {
        MPI_Waitall(request->local_S_n_msgs, request->local_S_requests,
                MPI_STATUSES_IGNORE);
        MPI_Reduce_local(request->tmpbuf, request->recvbuf, request->count,
                request->datatype, request->op);
    }

    for (int stride_start = 1; stride_start < max_proc; stride_start *= radix)
    {
        int ctr = 0;
        int n_msgs = 0;
        for (int step = 1; step < radix; step++)
        {
            int stride = stride_start * step;
            if (stride < max_proc)
                n_msgs += 2;
        }

        MPI_Startall(n_msgs, &(request->global_requests[ctr]));
        MPI_Waitall(n_msgs, &(request->global_requests[ctr]),
                MPI_STATUSES_IGNORE);

        for (int i = 0; i < n_msgs / 2; i++)
        {
            MPI_Reduce_local((char*)request->tmpbuf + i * request->count * type_size,
                    request->recvbuf, request->count, request->datatype, request->op);

        }
        ctr += n_msgs;
    }

    if (request->local_R_n_msgs)
    {
        MPI_Startall(request->local_R_n_msgs, request->local_R_requests);
        MPI_Waitall(request->local_R_n_msgs, request->local_R_requests,
                MPI_STATUSES_IGNORE);
    }
    
    return MPI_SUCCESS;
}
