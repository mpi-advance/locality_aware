#include "../../include/neighborhood/neighbor_init.h"

#include "../../include/neighborhood/neighbor.h"
#include "../../include/neighborhood/neighbor_persistent.h"

//enum NeighborAlltoallvInitMethod mpix_neighbor_alltoallv_init_implementation =
//    NEIGHBOR_ALLTOALLV_INIT_STANDARD;


int MPIL_Neighbor_alltoallv_init_ext(const void* sendbuf,
                                     const int sendcounts[],
                                     const int sdispls[],
                                     const long global_sindices[],
                                     MPI_Datatype sendtype,
                                     void* recvbuf,
                                     const int recvcounts[],
                                     const int rdispls[],
                                     const long global_rindices[],
                                     MPI_Datatype recvtype,
                                     MPIL_Comm* comm,
                                     MPIL_Info* info,
                                     MPIL_Request** request_ptr)
{
    MPIL_Topo* topo;
    MPIL_Topo_from_neighbor_comm(comm, &topo);

    MPIL_Neighbor_alltoallv_init_ext_topo(sendbuf,
                                          sendcounts,
                                          sdispls,
                                          global_sindices,
                                          sendtype,
                                          recvbuf,
                                          recvcounts,
                                          rdispls,
                                          global_rindices,
                                          recvtype,
                                          topo,
                                          comm,
                                          info,
                                          request_ptr);

    MPIL_Topo_free(&topo);

    return MPI_SUCCESS;
}
