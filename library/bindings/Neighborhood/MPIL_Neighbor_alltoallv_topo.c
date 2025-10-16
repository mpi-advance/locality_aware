#include "communicator/MPIL_Comm.h"
#include "locality_aware.h"
#include "neighborhood/neighbor.h"
// Standard Method is default

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

    switch (mpil_neighbor_alltoallv_implementation)
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
