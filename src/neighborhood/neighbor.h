#ifndef MPI_ADVANCE_NEIGHBOR_COLL_H
#define MPI_ADVANCE_NEIGHBOR_COLL_H

#include <mpi.h>
#include <stdlib.h>
#include "locality/locality_comm.h"
#include "dist_graph.h"
#include "persistent/persistent.h"

// Declarations of C++ methods
#ifdef __cplusplus
extern "C"
{
#endif


// Standard Persistent Neighbor Alltoallv
// Extension takes array of requests instead of single request
// 'requests' must be of size indegree+outdegree!
int MPIX_Neighbor_alltoallv_init(
        const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm,
        MPI_Info info,
        MPIX_Request** request_ptr);


/*
// Locality-Aware Extension to Persistent Neighbor Alltoallv
// Needs global indices for each send and receive
int MPIX_Neighbor_alltoallv_init(
        const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        const int global_sindices[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        const int global_rindices[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm,
        MPI_Info info,
        MPIX_Request** request_ptr);
*/








#ifdef __cplusplus
}
#endif

#endif
