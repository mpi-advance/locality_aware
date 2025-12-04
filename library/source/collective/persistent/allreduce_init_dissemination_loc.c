#include "collective/allreduce_init.h"
#include "locality_aware.h"
#include <string.h>
#include <math.h>

int allreduce_dissemination_loc_init(const void* sendbuf,
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

    int local_rank, ppn;
    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &ppn);

    int rank_node, num_nodes;
    MPI_Comm_rank(comm->group_comm, &rank_node);
    MPI_Comm_size(comm->group_comm, &num_nodes);

    int tag;
    get_tag(comm, &tag);

    // Locality-Aware only works if ppn is even on all processes
    if (num_nodes * ppn != num_procs)
        return allreduce_recursive_doubling_init_helper(
                sendbuf, recvbuf, count, datatype, op, comm,
                info, req_ptr, MPIL_Alloc, MPIL_Free);

    return allreduce_dissemination_loc_init_helper(sendbuf, recvbuf, count,
            datatype, op, comm->global_comm, comm->group_comm, 
            comm->local_comm, info, tag, req_ptr,
            MPIL_Alloc, MPIL_Free);
        
}



int allreduce_dissemination_ml_init(const void* sendbuf,
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

    int local_rank, ppn;
    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &ppn);

    int rank_node, num_nodes;
    MPI_Comm_rank(comm->group_comm, &rank_node);
    MPI_Comm_size(comm->group_comm, &num_nodes);

    int tag;
    get_tag(comm, &tag);

    // Locality-Aware only works if ppn is even on all processes
    if (num_nodes * ppn != num_procs)
        return allreduce_recursive_doubling_init_helper(
                sendbuf, recvbuf, count, datatype, op, comm,
                info, req_ptr, MPIL_Alloc, MPIL_Free);

    // Convert to leader_comm (4 leaders per node)
    int num_leaders = 4;
    if (comm->leader_comm != MPI_COMM_NULL)
    {
        int ppl;
        MPI_Comm_size(comm->leader_comm, &ppl);
        if (ppn / num_leaders != ppl)
        {
            MPIL_Comm_leader_free(comm);
        }
    }
    if (comm->leader_comm == MPI_COMM_NULL)
        MPIL_Comm_leader_init(comm, ppn / num_leaders);

    return allreduce_dissemination_loc_init_helper(
                   sendbuf, recvbuf, count, datatype, op,
                   comm->global_comm, comm->group_comm,
                   comm->local_comm, info, tag, req_ptr,
                   MPIL_Alloc, MPIL_Free);
}


int allreduce_dissemination_loc_init_helper(const void* sendbuf,
                                 void* recvbuf,
                                 int count,
                                 MPI_Datatype datatype,
                                 MPI_Op op,
                                 MPI_Comm global_comm, 
                                 MPI_Comm group_comm,
                                 MPI_Comm local_comm,
                                 MPIL_Info* info,
                                 int tag,
                                 MPIL_Request** req_ptr,
                                 MPIL_Alloc_ftn alloc_ftn,
                                 MPIL_Free_ftn free_ftn)
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
    int max_n_msgs = 2*(log2(num_procs));
    allocate_requests(max_n_msgs, &(request->global_requests));
    allocate_requests(1, &(request->local_L_requests));
    allocate_requests(1, &(request->local_S_requests));
    allocate_requests(2, &(request->local_R_requests));

    request->start_function = allreduce_dissemination_loc_start;
    request->wait_function  = allreduce_dissemination_loc_wait;

    MPI_Comm_dup(local_comm, &(request->local_comm));

    request->count = count;
    request->op = op;
    request->datatype = datatype;
    request->sendbuf = sendbuf;
    request->recvbuf = recvbuf;

    int type_size;
    MPI_Type_size(datatype, &type_size);

    alloc_ftn(&(request->tmpbuf), type_size*count);
    request->free_ftn = free_ftn;


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
                group_comm, &(request->local_L_requests[request->local_L_n_msgs++]));
        MPI_Recv_init(recvbuf, count, datatype, node, tag, 
                group_comm, &(request->local_R_requests[request->local_R_n_msgs++]));
    }
    else
    {
        if (rank_node < extra)
        {
            MPI_Recv_init(request->tmpbuf, count, datatype, max_node + rank_node, tag, 
                    group_comm, &(request->local_S_requests[request->local_S_n_msgs++]));
        }

        for (int node_stride = 1; node_stride < max_node; node_stride *= (ppn+1))
        {
            int stride = node_stride * (local_rank+1);
            if (stride < max_node)
            {
                int send_node = (rank_node - stride + max_node) % max_node;
                int recv_node = (rank_node + stride) % max_node;

                MPI_Send_init(recvbuf, count, datatype, send_node, tag, group_comm, 
                    &(request->global_requests[request->global_n_msgs++]));
                MPI_Recv_init(request->tmpbuf, count, datatype, recv_node, tag, group_comm, 
                    &(request->global_requests[request->global_n_msgs++]));
            }
            request->num_ops += 2;
        }

        if (rank_node < extra)
        {
            MPI_Send_init(recvbuf, count, datatype, max_node + rank_node, 
                    tag, group_comm, &(request->local_R_requests[request->local_R_n_msgs++]));
        }
    }

    *req_ptr = request;    

    return MPI_SUCCESS;
}


int allreduce_dissemination_loc_start(MPIL_Request* request)
{
    if (request == NULL)
        return 0;

    PMPI_Allreduce(request->sendbuf, request->recvbuf, request->count, 
            request->datatype, request->op, request->local_comm);

    if (request->local_L_n_msgs)
        MPI_Startall(request->local_L_n_msgs, request->local_L_requests);
    if (request->local_S_n_msgs)
        MPI_Startall(request->local_S_n_msgs, request->local_S_requests);

    return MPI_SUCCESS;
}

int allreduce_dissemination_loc_wait(MPIL_Request* request, MPI_Status* status)
{
    if (request == NULL)
        return 0;

    if (request->local_L_n_msgs)
        MPI_Waitall(request->local_L_n_msgs, request->local_L_requests, MPI_STATUSES_IGNORE);

    if (request->local_S_n_msgs)
    {
        MPI_Waitall(request->local_S_n_msgs, request->local_S_requests, MPI_STATUSES_IGNORE);
        MPI_Reduce_local(request->tmpbuf, request->recvbuf, request->count,
                request->datatype, request->op);
    }

    for (int i = 0; i < request->global_n_msgs; i += 2)
    {
        MPI_Startall(2, &(request->global_requests[i]));
        MPI_Waitall(2, &(request->global_requests[i]), MPI_STATUSES_IGNORE);
        MPI_Allreduce(MPI_IN_PLACE, request->tmpbuf, request->count,
                request->datatype, request->op, request->local_comm);
        MPI_Reduce_local(request->tmpbuf, request->recvbuf, request->count,
                request->datatype, request->op);
    }
    if (request->num_ops > request->global_n_msgs)
    {
        int type_size;
        MPI_Type_size(request->datatype, &type_size);
        memset(request->tmpbuf, 0, request->count * type_size);
        MPI_Allreduce(MPI_IN_PLACE, request->tmpbuf, request->count,
                request->datatype, request->op, request->local_comm);
        MPI_Reduce_local(request->tmpbuf, request->recvbuf, request->count,
                request->datatype, request->op);
    }

    if (request->local_R_n_msgs)
    {
        MPI_Startall(request->local_R_n_msgs, request->local_R_requests);
        MPI_Waitall(request->local_R_n_msgs, request->local_R_requests, MPI_STATUSES_IGNORE);
    }
    
    return MPI_SUCCESS;
}

