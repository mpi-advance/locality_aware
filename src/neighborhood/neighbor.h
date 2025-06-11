#ifndef MPI_ADVANCE_NEIGHBOR_COLL_H
#define MPI_ADVANCE_NEIGHBOR_COLL_H

#include <mpi.h>
#include <stdlib.h>
#include "dist_graph.h"
#include "dist_topo.h"
#include "persistent/persistent.h"
#include "locality/locality_comm.h"
#include "utils/utils.h"

// Declarations of C++ methods
#ifdef __cplusplus
extern "C"
{
#endif

// Standard Neighbor Alltoallv
// Extension takes array of requests instead of single request
// 'requests' must be of size indegree+outdegree!
int MPIX_Neighbor_topo_alltoallv(
        const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Topo* topo,
        MPIX_Comm* comm);

int MPIX_Neighbor_alltoallv(
        const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm);


#ifdef __cplusplus
}
#endif

#endif
