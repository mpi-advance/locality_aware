#ifndef MPI_ADVANCE_NEIGHBOR_PERSISTENT_H
#define MPI_ADVANCE_NEIGHBOR_PERSISTENT_H


#include "neighbor.h"
#include "../persistent/persistent.h"

#ifdef __cplusplus
extern "C" {
#endif

enum NeighborAlltoallvInitMethod
{
    NEIGHBOR_ALLTOALLV_INIT_STANDARD,
    NEIGHBOR_ALLTOALLV_INIT_LOCALITY
};
extern enum NeighborAlltoallvInitMethod mpix_neighbor_alltoallv_init_implementation;
void MPIL_set_alltoallv_neighbor_init_alogorithm(enum NeighborAlltoallvInitMethod algorithm);

typedef int (*neighbor_alltoallv_init_ftn)(const void* sendbuf,
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
                                           MPIL_Request** request_ptr);


// Standard Persistent Neighbor Alltoallv
// Extension takes array of requests instead of single request
// 'requests' must be of size indegree+outdegree!
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
                                      MPIL_Request** request_ptr);
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
                                 MPIL_Request** request_ptr);

// Locality-Aware Extension to Persistent Neighbor Alltoallv
// Needs global indices for each send and receive
int MPIL_Neighbor_alltoallv_init_ext_topo(const void* sendbuf,
                                          const int sendcounts[],
                                          const int sdispls[],
                                          const long global_sindices[],
                                          MPI_Datatype sendtype,
                                          void* recvbuf,
                                          const int recvcounts[],
                                          const int rdispls[],
                                          const long global_rindices[],
                                          MPI_Datatype recvtype,
                                          MPIL_Topo* topo,
                                          MPIL_Comm* comm,
                                          MPIL_Info* info,
                                          MPIL_Request** request_ptr);
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
                                     MPIL_Request** request_ptr);

#ifdef __cplusplus
}
#endif

#endif
