#include "locality_aware.h"


// Standard Persistent Neighbor Alltoallv


// Standard Persistent Neighbor Alltoallv
// Extension takes array of requests instead of single request
// 'requests' must be of size indegree+outdegree!
int MPIL_Neighbor_alltoallv_init(const void* sendbuf,
                                 const int sendcounts[],
                                 const int sdispls[],
                                 MPI_Datatype sendtype,
                                 void* recvbuf,
                                 const int recvcounts[],
                                 const int rdispls[],
                                 MPI_Datatype recvtype,
                                 MPIL_Comm* comm,
                                 MPIL_Info* info,
                                 MPIL_Request** request_ptr)
{
    MPIL_Topo* topo;
    MPIL_Topo_from_neighbor_comm(comm, &topo);

    MPIL_Neighbor_alltoallv_init_topo(sendbuf,
                                      sendcounts,
                                      sdispls,
                                      sendtype,
                                      recvbuf,
                                      recvcounts,
                                      rdispls,
                                      recvtype,
                                      topo,
                                      comm,
                                      info,
                                      request_ptr);

    MPIL_Topo_free(&topo);

    return MPI_SUCCESS;
}