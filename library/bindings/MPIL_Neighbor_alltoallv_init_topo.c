#include "locality_aware.h"
#include "../include/neighborhood/neighborhood_init.h"


int MPIL_Neighbor_alltoallv_init_topo(const void* sendbuf,
                                      const int sendcounts[],
                                      const int sdispls[],
                                      MPI_Datatype sendtype,
                                      void* recvbuf,
                                      const int recvcounts[],
                                      const int rdispls[],
                                      MPI_Datatype recvtype,
                                      MPIL_Topo* topo,
                                      MPIL_Comm* comm,
                                      MPIL_Info* info,
                                      MPIL_Request** request_ptr)
{
    neighbor_alltoallv_init_ftn method;

    switch (mpix_neighbor_alltoallv_init_implementation)
    {
        case NEIGHBOR_ALLTOALLV_INIT_STANDARD:
            method = neighbor_alltoallv_init_standard;
            break;
        case NEIGHBOR_ALLTOALLV_INIT_LOCALITY:
            method = neighbor_alltoallv_init_locality;
            break;
        default:
            method = neighbor_alltoallv_init_standard;
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
                  comm,
                  info,
                  request_ptr);
}
