#include "collective/allreduce.hpp"
#include "locality_aware.h"
#include <string.h>
#include <math.h>
#include <stdio.h>

// TODO: fix memset to allow for gpus
// Warning: assumes even numbers of processes per node
int allreduce_dissemination_loc(const void* sendbuf,
                                 void* recvbuf,
                                 int count,
                                 MPI_Datatype datatype,
                                 MPI_Op op,
                                 MPIL_Comm* comm)
{
    return allreduce_impl(allreduce_dissemination_loc_helper,
                   sendbuf, recvbuf, count, datatype, op, comm,
                   MPIL_Alloc, MPIL_Free);
}

int allreduce_dissemination_loc_helper(
                        const void* sendbuf,
                        void* tmpbuf,
                        void* recvbuf,
                        int count,
                        MPI_Datatype datatype,
                        MPI_Op op,
                        MPIL_Comm* comm)
{
    if (count == 0)
        return MPI_SUCCESS;

    int type_size;
    MPI_Type_size(datatype, &type_size);

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

    // Locality-Aware only works if ppn is even on all processes
    if (num_nodes * ppn != num_procs)
        return allreduce_recursive_doubling_helper(
                sendbuf, tmpbuf, recvbuf, count, datatype, op, comm);

    int tag;
    get_tag(comm, &tag);

    PMPI_Allreduce(sendbuf, recvbuf, count, datatype,
            op, comm->local_comm);

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
        MPI_Send(recvbuf, count, datatype, node, tag, comm->group_comm);
        MPI_Recv(recvbuf, count, datatype, node, tag, comm->group_comm,
                MPI_STATUS_IGNORE);
    }
    else
    {
        if (rank_node < extra)
        {
            MPI_Recv(tmpbuf, count, datatype, max_node + rank_node, tag, comm->group_comm,
                   MPI_STATUS_IGNORE);
            MPI_Reduce_local(tmpbuf, recvbuf, count, datatype, op);
        }

        for (int node_stride = 1; node_stride < max_node; node_stride *= (ppn+1))
        {
            int stride = node_stride * (local_rank+1);
            if (stride < max_node)
            {
                int send_node = (rank_node - stride + max_node) % max_node;
                int recv_node = (rank_node + stride) % max_node;
                MPI_Sendrecv(recvbuf, count, datatype, send_node, tag,
                        tmpbuf, count, datatype, recv_node, tag,
                        comm->group_comm, MPI_STATUS_IGNORE);
            }
            else
                memset(tmpbuf, 0, count*type_size);
            MPI_Allreduce(MPI_IN_PLACE, tmpbuf, count, datatype, op, comm->local_comm);
            MPI_Reduce_local(tmpbuf, recvbuf, count, datatype, op);
        }

        if (rank_node < extra)
        {
            MPI_Send(recvbuf, count, datatype, max_node + rank_node, tag, comm->group_comm);
        }
    }

    return MPI_SUCCESS;
}


