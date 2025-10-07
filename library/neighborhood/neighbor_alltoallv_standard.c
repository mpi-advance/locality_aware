#include "../../include/neighborhood/neighbor.h"

#ifdef __cplusplus
#include <cstring>
#endif

#ifndef __cplusplus
#include "string.h"
#endif

#include "../../include/neighborhood/sparse_coll.h"


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

