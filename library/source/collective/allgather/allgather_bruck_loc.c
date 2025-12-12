#include "collective/allreduce.h"
#include "locality_aware.h"
#include <string.h>
#include <math.h>
#include <stdio.h>

// TODO: fix memset to allow for gpus
// Warning: assumes even numbers of processes per node
int allgather_bruck_loc(const void* sendbuf,
                        int sendcount,
                        MPI_Datatype sendtype,
                        void* recvbuf,
                        int recvcount,
                        MPI_Datatype recvtype,
                        MPIL_Comm* comm)
{
    if (sendcount == 0)
        return MPI_SUCCESS;

    return allgather_bruck_loc_helper(sendbuf, sendcount, sendtype,
            recvbuf, recvcount, recvtype, comm, MPIL_Alloc, MPIL_Free);
}

int allgather_bruck_ml(const void* sendbuf,
                        int sendcount,
                        MPI_Datatype sendtype,
                        void* recvbuf,
                        int recvcount,
                        MPI_Datatype recvtype,
                        MPIL_Comm* comm)
{
    if (sendcount == 0)
        return MPI_SUCCESS;

    return allgather_bruck_ml_helper(sendbuf, sendcount, sendtype,
            recvbuf, recvcount, recvtype, comm, MPIL_Alloc, MPIL_Free);
}

int allgather_bruck_loc_helper(const void* sendbuf,
                               int sendcount,
                               MPI_Datatype sendtype,
                               void* recvbuf,
                               int recvcount,
                               MPI_Datatype recvtype,
                               MPIL_Comm* comm,
                               MPIL_Alloc_ftn alloc_ftn, 
                               MPIL_Free_ftn free_ftn)
{
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
        return allgather_bruck_helper(sendbuf, sendcount, sendtype,
            recvbuf, recvcount, recvtype, comm, MPIL_Alloc, MPIL_Free);

    return allgather_bruck_loc_core(sendbuf, sendcount, sendtype,
                        recvbuf, recvcount, recvtype, comm->global_comm,
                        comm->group_comm, comm->local_comm, tag, 
                        alloc_ftn, free_ftn);
}


int allgather_bruck_ml_helper(const void* sendbuf,
                               int sendcount,
                               MPI_Datatype sendtype,
                               void* recvbuf,
                               int recvcount,
                               MPI_Datatype recvtype,
                               MPIL_Comm* comm,
                               MPIL_Alloc_ftn alloc_ftn, 
                               MPIL_Free_ftn free_ftn)
{
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
        return allgather_bruck_helper(sendbuf, sendcount, sendtype,
            recvbuf, recvcount, recvtype, comm, MPIL_Alloc, MPIL_Free);

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

    return allgather_bruck_loc_core(sendbuf, sendcount, sendtype,
                        recvbuf, recvcount, recvtype, comm->global_comm,
                        comm->leader_group_comm, comm->leader_comm, tag, 
                        alloc_ftn, free_ftn);
}

int allgather_bruck_loc_core(
                        const void* sendbuf,
                        int sendcount, 
                        MPI_Datatype sendtype,
                        void* recvbuf,
                        int recvcount, 
                        MPI_Datatype recvtype,
                        MPI_Comm global_comm, 
                        MPI_Comm group_comm,
                        MPI_Comm local_comm,
                        int tag,
                        MPIL_Alloc_ftn alloc_ftn,
                        MPIL_Free_ftn free_ftn)
{
    int type_size;
    MPI_Type_size(datatype, &type_size);

    int rank, num_procs;
    MPI_Comm_rank(global_comm, &rank);
    MPI_Comm_size(global_comm, &num_procs);

    int local_rank, ppn;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &ppn);

    int rank_node, num_nodes;
    MPI_Comm_rank(group_comm, &rank_node);
    MPI_Comm_size(group_comm, &num_nodes);

    int pow_ppn_num_nodes = 1;
    int base = ppn + 1;
    while (pow_ppn_num_nodes * base <= num_nodes)
        pow_ppn_num_nodes *= base;
    int mult = num_nodes / pow_ppn_num_nodes;
    int max_node = mult * pow_ppn_num_nodes;
    int extra = num_nodes - max_node;

    void *tmpbuf;
    alloc_ftn(&tmpbuf, type_size*count);

    PMPI_Allgather(sendbuf, sendcount, sendtype, tmpbuf, recvcount, recvtype,
            local_comm);


    // Main loop of Locality-Aware Bruck
    int size = ppn * recvcount;
    int node_stride;
    for (node_stride = 1; node_stride < max_node; node_stride *= (ppn+1))
    {
        int stride = node_stride * (local_rank+1);
        if (stride < max_node)
        {
            int send_node = (rank_node - stride + num_nodes) % num_nodes;
            int recv_node = (rank_node + stride) % num_nodes;
            MPI_Sendrecv(tmpbuf, size, recvtype, send_node, tag,
                    tmpbuf + size * type_size, size, recvtype, recv_node, tag,
                    group_comm, MPI_STATUS_IGNORE);
        }

        // Do an allgather until the last iteration, in which all ranks may
        // not be active
        if (node_stride * ppn < max_node)
            PMPI_Allgather(MPI_IN_PLACE, size, recvtype, tmpbuf + size * type_size,
                    size, recvtype, local_comm);
        else // If some ranks have no buffer to contribute, do allgatherv
            PMPI_Allgatherv();
        size *= ppn;
    }

    if (extra)
    {
        int stride = node_stride * (local_rank+1);
        if (stride < num_nodes)
        {
            int size = extra * ppn * recvcount;
            int send_node = (rank_node - stride + max_node) % max_node;
            int recv_node = (rank_node + stride) % max_node;
            MPI_Sendrecv(tmpbuf, size, recvtype, send_node, tag,
                    tmpbuf + size * type_size, size, recvtype, recv_node, tag,
                    group_comm, MPI_STATUS_IGNORE);
        }
        PMPI_Allgather(MPI_IN_PLACE, size, recvtype, tmpbuf + size * type_size,
                size, recvtype, local_comm);
        size *= ppn;
    }



    free_ftn(tmpbuf);

    return MPI_SUCCESS;
}


