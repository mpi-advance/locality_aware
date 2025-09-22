#include "neighbor.h"

#include <cstring>

#include "sparse_coll.h"

// Standard Method is default
NeighborAlltoallvMethod mpix_neighbor_alltoallv_implementation =
    NEIGHBOR_ALLTOALLV_STANDARD;

// Topology object based neighbor alltoallv
int MPIL_Neighbor_alltoallv_topo(const void* sendbuf,
                                 const int sendcounts[],
                                 const int sdispls[],
                                 MPI_Datatype sendtype,
                                 void* recvbuf,
                                 const int recvcounts[],
                                 const int rdispls[],
                                 MPI_Datatype recvtype,
                                 MPIL_Topo* topo,
                                 MPIL_Comm* comm)
{
    int rank;
    MPI_Comm_rank(comm->global_comm, &rank);

    neighbor_alltoallv_ftn method;

    switch (mpix_neighbor_alltoallv_implementation)
    {
        case NEIGHBOR_ALLTOALLV_STANDARD:
            method = neighbor_alltoallv_standard;
            break;
        case NEIGHBOR_ALLTOALLV_LOCALITY:
            method = neighbor_alltoallv_locality;
            break;
        default:
            method = neighbor_alltoallv_standard;
            break;
    }

    return method(sendbuf,
                  sendcounts,
                  sdispls,
                  sendtype,
                  recvbuf,
                  recvcounts,
                  rdispls,
                  recvtype,
                  topo,
                  comm);
}

int MPIL_Neighbor_alltoallv(const void* sendbuffer,
                            const int sendcounts[],
                            const int sdispls[],
                            MPI_Datatype sendtype,
                            void* recvbuffer,
                            const int recvcounts[],
                            const int rdispls[],
                            MPI_Datatype recvtype,
                            MPIL_Comm* comm)
{
    MPIL_Topo* topo;
    MPIL_Topo_from_neighbor_comm(comm, &topo);

    MPIL_Neighbor_alltoallv_topo(sendbuffer,
                                 sendcounts,
                                 sdispls,
                                 sendtype,
                                 recvbuffer,
                                 recvcounts,
                                 rdispls,
                                 recvtype,
                                 topo,
                                 comm);

    MPIL_Topo_free(&topo);

    return MPI_SUCCESS;
}

// Standard, non-persistent neighbor collective
int neighbor_alltoallv_standard(const void* sendbuf,
                                const int sendcounts[],
                                const int sdispls[],
                                MPI_Datatype sendtype,
                                void* recvbuf,
                                const int recvcounts[],
                                const int rdispls[],
                                MPI_Datatype recvtype,
                                MPIL_Topo* topo,
                                MPIL_Comm* comm)
{
    int tag;
    MPIL_Comm_tag(comm, &tag);

    if (topo->indegree + topo->outdegree == 0)
    {
        return MPI_SUCCESS;
    }

    if (comm->n_requests < topo->indegree + topo->outdegree)
    {
        MPIL_Comm_req_resize(comm, topo->indegree + topo->outdegree);
    }

    const char* send_buffer = NULL;
    char* recv_buffer       = NULL;

    if (topo->indegree)
    {
        recv_buffer = (char*)recvbuf;
    }
    if (topo->outdegree)
    {
        send_buffer = (char*)sendbuf;
    }

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    int count = 0;
    for (int i = 0; i < topo->indegree; i++)
    {
        if (recvcounts[i])
        {
            MPI_Irecv(&(recv_buffer[rdispls[i] * recv_size]),
                      recvcounts[i],
                      recvtype,
                      topo->sources[i],
                      tag,
                      comm->global_comm,
                      &(comm->requests[count++]));
        }
    }

    for (int i = 0; i < topo->outdegree; i++)
    {
        if (sendcounts[i])
        {
            MPI_Isend(&(send_buffer[sdispls[i] * send_size]),
                      sendcounts[i],
                      sendtype,
                      topo->destinations[i],
                      tag,
                      comm->global_comm,
                      &(comm->requests[count++]));
        }
    }

    MPI_Waitall(count, comm->requests, comm->statuses);

    return MPI_SUCCESS;
}

// Non-persistent, locality-aware == call dynamic version
int neighbor_alltoallv_locality(const void* sendbuf,
                                const int sendcounts[],
                                const int sdispls[],
                                MPI_Datatype sendtype,
                                void* recvbuf,
                                const int recvcounts[],
                                const int rdispls[],
                                MPI_Datatype recvtype,
                                MPIL_Topo* topo,
                                MPIL_Comm* comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int send_nnz  = topo->outdegree;
    int send_size = 0;
    for (int i = 0; i < send_nnz; i++)
    {
        send_size += sendcounts[i];
    }

    int recv_nnz, recv_size;
    int *src_tmp, *recvcounts_tmp, *rdispls_tmp;
    char* recvvals_tmp;

    int recv_bytes;
    MPI_Type_size(recvtype, &recv_bytes);

    MPIL_Info* xinfo;
    MPIL_Info_init(&xinfo);

    alltoallv_crs_personalized(send_nnz,
                               send_size,
                               topo->destinations,
                               sendcounts,
                               sdispls,
                               sendtype,
                               sendbuf,
                               &recv_nnz,
                               &recv_size,
                               &src_tmp,
                               &recvcounts_tmp,
                               &rdispls_tmp,
                               recvtype,
                               (void**)&recvvals_tmp,
                               xinfo,
                               comm);

    char* recvvals = (char*)recvbuf;

    int idx, proc;
    int* new_proc_idx = (int*)malloc(num_procs * sizeof(int));
    for (int i = 0; i < recv_nnz; i++)
    {
        new_proc_idx[src_tmp[i]] = i;
    }
    for (int i = 0; i < recv_nnz; i++)
    {
        proc = topo->sources[i];
        idx  = new_proc_idx[proc];
        memcpy(&(recvvals[rdispls[i] * recv_bytes]),
               &(recvvals_tmp[rdispls_tmp[idx] * recv_bytes]),
               recvcounts[i] * recv_bytes);
    }
    free(new_proc_idx);

    MPIL_Info_free(&xinfo);

    MPIL_Free(src_tmp);
    MPIL_Free(recvcounts_tmp);
    MPIL_Free(rdispls_tmp);
    MPIL_Free(recvvals_tmp);

    return MPI_SUCCESS;
}
