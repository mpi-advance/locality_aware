#include <stdlib.h>

#include "communicator/MPIL_Comm.h"
#include "locality_aware.h"  // #TODO -- figure out why this is needed
#include "neighborhood/MPIL_Topo.h"
#include "neighborhood/neighborhood_init.h"

int neighbor_alltoallv_init_locality(const void* sendbuffer,
                                     const int sendcounts[],
                                     const int sdispls[],
                                     MPI_Datatype sendtype,
                                     void* recvbuffer,
                                     const int recvcounts[],
                                     const int rdispls[],
                                     MPI_Datatype recvtype,
                                     MPIL_Topo* topo,
                                     MPIL_Comm* comm,
                                     MPIL_Info* info,
                                     MPIL_Request** request_ptr)
{
    int rank;
    MPI_Comm_rank(comm->global_comm, &rank);

    if (comm->local_comm == MPI_COMM_NULL)
    {
        MPIL_Comm_topo_init(comm);
    }

    int* global_sdispls = NULL;
    int* global_rdispls = NULL;

    int ctr;

    if (topo->indegree)
    {
        global_rdispls = (int*)malloc(topo->indegree * sizeof(int));
        ctr            = 0;
        for (int i = 0; i < topo->indegree; i++)
        {
            global_rdispls[i] = ctr;
            ctr += recvcounts[i];
        }
    }

    if (topo->outdegree)
    {
        global_sdispls = (int*)malloc(topo->outdegree * sizeof(int));
        ctr            = 0;
        for (int i = 0; i < topo->outdegree; i++)
        {
            global_sdispls[i] = ctr;
            ctr += sendcounts[i];
        }
    }

    long send_size = 0;
    long recv_size = 0;
    for (int i = 0; i < topo->indegree; i++)
    {
        recv_size += recvcounts[i];
    }
    for (int i = 0; i < topo->outdegree; i++)
    {
        send_size += sendcounts[i];
    }

    long first_send;
    MPI_Exscan(&send_size, &first_send, 1, MPI_LONG, MPI_SUM, comm->global_comm);
    if (rank == 0)
    {
        first_send = 0;
    }

    long* global_send_indices = NULL;
    long* global_recv_indices = NULL;

    if (recv_size)
    {
        global_recv_indices = (long*)malloc(recv_size * sizeof(long));
    }
    if (send_size)
    {
        global_send_indices = (long*)malloc(send_size * sizeof(long));
    }
    for (int i = 0; i < send_size; i++)
    {
        global_send_indices[i] = first_send + i;
    }

    MPIL_Neighbor_alltoallv_topo(global_send_indices,
                                 sendcounts,
                                 global_sdispls,
                                 MPI_LONG,
                                 global_recv_indices,
                                 recvcounts,
                                 global_rdispls,
                                 MPI_LONG,
                                 topo,
                                 comm);

    int err = neighbor_alltoallv_init_locality_ext(sendbuffer,
                                                   sendcounts,
                                                   sdispls,
                                                   global_send_indices,
                                                   sendtype,
                                                   recvbuffer,
                                                   recvcounts,
                                                   rdispls,
                                                   global_recv_indices,
                                                   recvtype,
                                                   topo,
                                                   comm,
                                                   info,
                                                   request_ptr);

    free(global_send_indices);
    free(global_recv_indices);

    free(global_sdispls);
    free(global_rdispls);

    return err;
}
