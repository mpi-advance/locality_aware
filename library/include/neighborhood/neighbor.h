#ifndef MPI_ADVANCE_NEIGHBOR_COLL_H
#define MPI_ADVANCE_NEIGHBOR_COLL_H

#include "neighborhood/MPIL_Topo.h"
#include "communicator/MPIL_Comm.h"

// Declarations of C++ methods
#ifdef __cplusplus
extern "C" {
#endif

typedef int (*neighbor_alltoallv_ftn)(const void* sendbuffer,
                                      const int sendcounts[],
                                      const int sdispls[],
                                      MPI_Datatype sendtype,
                                      void* recvbuf,
                                      const int recvcounts[],
                                      const int rdispls[],
                                      MPI_Datatype recvtype,
                                      MPIL_Topo* topo,
                                      MPIL_Comm* comm);

// Standard Neighbor Alltoallv
// Extension takes array of requests instead of single request
// 'requests' must be of size indegree+outdegree!

int neighbor_alltoallv_standard(const void* sendbuf,
                                const int sendcounts[],
                                const int sdispls[],
                                MPI_Datatype sendtype,
                                void* recvbuf,
                                const int recvcounts[],
                                const int rdispls[],
                                MPI_Datatype recvtype,
                                MPIL_Topo* topo,
                                MPIL_Comm* comm);

int neighbor_alltoallv_locality(const void* sendbuf,
                                const int sendcounts[],
                                const int sdispls[],
                                MPI_Datatype sendtype,
                                void* recvbuf,
                                const int recvcounts[],
                                const int rdispls[],
                                MPI_Datatype recvtype,
                                MPIL_Topo* topo,
                                MPIL_Comm* comm);

#ifdef __cplusplus
}
#endif

#endif
