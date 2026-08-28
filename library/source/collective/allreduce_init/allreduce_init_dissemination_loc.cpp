#include "collective/allreduce_init.h"
#include "heterogeneous/gpu_utils.h"
#include "locality_aware.h"
#include <string.h>
#include <math.h>

int allreduce_init_dissemination_loc(const void* sendbuf,
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

    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    if (comm->local_comm == MPI_COMM_NULL)
        MPIL_Comm_topo_init(comm);

    int ppn;
    MPI_Comm_size(comm->local_comm, &ppn);

    int num_nodes;
    MPI_Comm_size(comm->group_comm, &num_nodes);

    int tag;
    get_tag(comm, &tag);

    // Locality-Aware only works if ppn is even on all processes
    if (num_nodes * ppn != num_procs)
    {
        return allreduce_init_recursive_doubling(
                sendbuf, recvbuf, count, datatype, op, comm,
                info, req_ptr);
    }

    return allreduce_init_dissemination_loc_core(sendbuf, recvbuf, count,
            datatype, op, comm->global_comm, comm->group_comm, 
            comm->local_comm, tag, info, req_ptr);
        
}

int allreduce_init_dissemination_ml(const void* sendbuf,
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
    
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);
    
    if (comm->local_comm == MPI_COMM_NULL)
        MPIL_Comm_topo_init(comm);
    
    int ppn;
    MPI_Comm_size(comm->local_comm, &ppn);

    int num_nodes;
    MPI_Comm_size(comm->group_comm, &num_nodes);

    int tag;
    get_tag(comm, &tag);

    // Locality-Aware only works if ppn is even on all processes
    if (num_nodes * ppn != num_procs)
    {
        return allreduce_init_recursive_doubling(
                sendbuf, recvbuf, count, datatype, op, comm,
                info, req_ptr);
    }

    // Convert to le/ader_comm (4 leaders per node)
    int num_leaders = 4;
    if (ppn < num_leaders)
    {
        num_leaders = ppn;
    }

    int ppl;
    if (comm->leader_comm != MPI_COMM_NULL)
    {   
        MPI_Comm_size(comm->leader_comm, &ppl);
    }

    if (comm->leader_comm == MPI_COMM_NULL || ppn / num_leaders != ppl)
    {
        MPIL_Comm_leader_init(comm, ppn/num_leaders);
    }

    return allreduce_init_dissemination_loc_core(
                   sendbuf, recvbuf, count, datatype, op,
                   comm->global_comm, comm->leader_group_comm,
                   comm->leader_comm, tag, info, req_ptr);
}


int allreduce_init_dissemination_loc_core(const void* sendbuf,
                                 void* recvbuf,
                                 int count,
                                 MPI_Datatype datatype,
                                 MPI_Op op,
                                 MPI_Comm global_comm, 
                                 Communicator::CachedComm group_comm,
                                 Communicator::CachedComm local_comm,
                                 int tag,
                                 MPIL_Info* info,
                                 MPIL_Request** req_ptr)
{

    int rank, num_procs;
    MPI_Comm_rank(global_comm, &rank);
    MPI_Comm_size(global_comm, &num_procs);

    int local_rank, ppn;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &ppn);

    int rank_node, num_nodes;
    MPI_Comm_rank(group_comm, &rank_node);
    MPI_Comm_size(group_comm, &num_nodes);

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
    allocate_requests(1, local_L_request);
    allocate_requests(1, local_S_request);
    allocate_requests(1, local_R_request);
    request->n_msgs = 0;
    local_L_request->n_msgs = 0;
    local_S_request->n_msgs = 0;
    local_R_request->n_msgs = 0;

    request->start_function = allreduce_dissemination_loc_start;
    request->wait_function  = allreduce_dissemination_loc_wait;

    request->local_comm = local_comm;

    request->count = count;
    request->op = op;
    request->datatype = datatype;
    request->sendbuf = sendbuf;
    request->recvbuf = recvbuf;

    int type_size;
    MPI_Type_size(datatype, &type_size);

    MPIL_Alloc(&(request->tmpbuf), type_size*count);

    int pow_ppn_num_nodes = 1;
    int base = ppn + 1;
    while (pow_ppn_num_nodes * base <= num_nodes)
        pow_ppn_num_nodes *= base;
    int mult = num_nodes / pow_ppn_num_nodes;
    int max_node = mult * pow_ppn_num_nodes;
    int extra = num_nodes - max_node;

    if (rank_node >= max_node)
    {
        int node = rank_node - max_node;
        MPI_Send_init(recvbuf, count, datatype, node, tag, 
                group_comm, &(local_L_request->requests[local_L_request->n_msgs++]));
        MPI_Recv_init(recvbuf, count, datatype, node, tag, 
                group_comm, &(local_R_request->requests[local_R_request->n_msgs++]));
    }
    else
    {
        if (rank_node < extra)
        {
            MPI_Recv_init(request->tmpbuf, count, datatype, max_node + rank_node, tag, 
                    group_comm, &(local_S_request->requests[local_S_request->n_msgs++]));
        }

        for (int node_stride = 1; node_stride < max_node; node_stride *= (ppn+1))
        {
            int stride = node_stride * (local_rank+1);
            if (stride < max_node)
            {
                int send_node = (rank_node - stride + max_node) % max_node;
                int recv_node = (rank_node + stride) % max_node;

                MPI_Send_init(recvbuf, count, datatype, send_node, tag, group_comm, 
                    &(request->requests[request->n_msgs++]));
                MPI_Recv_init(request->tmpbuf, count, datatype, recv_node, tag, group_comm, 
                    &(request->requests[request->n_msgs++]));
            }
            request->num_ops += 2;
        }

        if (rank_node < extra)
        {
            MPI_Send_init(recvbuf, count, datatype, max_node + rank_node, 
                    tag, group_comm, &(local_R_request->requests[local_R_request->n_msgs++]));
        }
    }

    *req_ptr = request;    

    return MPI_SUCCESS;
}


int allreduce_dissemination_loc_start(MPIL_Request* request)
{
    if (request == NULL)
    {
        return MPI_SUCCESS;
    }

    MPIL_Request* local_L_request = request->local_L_request;
    MPIL_Request* local_S_request = request->local_S_request;
    MPIL_Request* local_R_request = request->local_R_request;

    int type_size;
    MPI_Type_size(request->datatype, &type_size);

#if defined(GPU)
int gpu_error;
if (request->gpu_sendbuf)
{
// tmp_sendbuf is same as sendbuf, but not const
#if defined(APU)
    memcpy(request->tmp_gpubuf, request->gpu_sendbuf, request->size_recvs);
#else
    gpu_error = gpuMemcpyAsync(request->tmp_gpubuf, request->gpu_sendbuf, 
            request->size_recvs, gpuMemcpyDeviceToHost, 0);
    gpu_check(gpu_error);
    gpu_error = gpuStreamSynchronize(0);
    gpu_check(gpu_error);
#endif
}
#endif

    if (request == NULL)
        return 0;

    PMPI_Allreduce(request->sendbuf, request->recvbuf, request->count, 
            request->datatype, request->op, request->local_comm);

    if (local_L_request->n_msgs)
    {
        MPI_Startall(local_L_request->n_msgs, local_L_request->requests);
    }

    if (local_S_request->n_msgs)
    {
        MPI_Startall(local_S_request->n_msgs, local_S_request->requests);
    }

    return MPI_SUCCESS;
}

int allreduce_dissemination_loc_wait(MPIL_Request* request, MPI_Status* status)
{
    MPIL_Request* local_L_request = request->local_L_request;
    MPIL_Request* local_S_request = request->local_S_request;
    MPIL_Request* local_R_request = request->local_R_request;

    int type_size;
    MPI_Type_size(request->datatype, &type_size);

    if (request == NULL)
        return 0;

    if (local_L_request->n_msgs)
    {
        MPI_Waitall(local_L_request->n_msgs, local_L_request->requests, MPI_STATUSES_IGNORE);
    }

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
        MPI_Allreduce(MPI_IN_PLACE, request->tmpbuf, request->count,
                request->datatype, request->op, request->local_comm);
        MPI_Reduce_local(request->tmpbuf, request->recvbuf, request->count,
                request->datatype, request->op);
    }
    if (request->num_ops > request->n_msgs)
    {
        int type_size;
        MPI_Type_size(request->datatype, &type_size);
        memset(request->tmpbuf, 0, request->count * type_size);
        MPI_Allreduce(MPI_IN_PLACE, request->tmpbuf, request->count,
                request->datatype, request->op, request->local_comm);
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
    int gpu_error;
#if defined(APU)
    memcpy(request->gpu_recvbuf, request->recvbuf, request->size_recvs);
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


