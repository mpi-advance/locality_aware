#ifndef MPI_ADVANCE_NEIGHBOR_COLL_H
#define MPI_ADVANCE_NEIGHBOR_COLL_H

#include <mpi.h>
#include <stdlib.h>


#include "MPIL_Graph.h"
#include "MPIL_Topo.h"
#include "../persistent/persistent.h"


// Declarations of C++ methods
#ifdef __cplusplus
extern "C" {
#endif

enum NeighborAlltoallvMethod
{
    NEIGHBOR_ALLTOALLV_STANDARD,
    NEIGHBOR_ALLTOALLV_LOCALITY
};
extern enum NeighborAlltoallvMethod mpix_neighbor_alltoallv_implementation;
void MPIL_set_alltoall_neighbor_alogorithm(enum NeighborAlltoallvMethod algorithm);


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
int MPIL_Neighbor_alltoallv_topo(const void* sendbuf,
                                 const int sendcounts[],
                                 const int sdispls[],
                                 MPI_Datatype sendtype,
                                 void* recvbuf,
                                 const int recvcounts[],
                                 const int rdispls[],
                                 MPI_Datatype recvtype,
                                 MPIL_Topo* topo,
                                 MPIL_Comm* comm);

int MPIL_Neighbor_alltoallv(const void* sendbuf,
                            const int sendcounts[],
                            const int sdispls[],
                            MPI_Datatype sendtype,
                            void* recvbuf,
                            const int recvcounts[],
                            const int rdispls[],
                            MPI_Datatype recvtype,
                            MPIL_Comm* comm);

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
