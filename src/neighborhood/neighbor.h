#ifndef MPI_ADVANCE_NEIGHBOR_COLL_H
#define MPI_ADVANCE_NEIGHBOR_COLL_H

#include <mpi.h>
#include <stdlib.h>
#include "dist_graph.h"
#include "persistent/persistent.h"
#include "locality/locality_comm.h"

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


// Standard Persistent Neighbor Alltoallv
// Extension takes array of requests instead of single request
// 'requests' must be of size indegree+outdegree!
int MPIX_Neighbor_alltoallw_init(
        const void* sendbuf,
        const int sendcounts[],
        const MPI_Aint sdispls[],
        MPI_Datatype* sendtypes,
        void* recvbuf,
        const int recvcounts[],
        const MPI_Aint rdispls[],
        MPI_Datatype* recvtypes,
        MPIX_Comm* comm,
        MPI_Info info,
        MPIX_Request** request_ptr);



// Locality-Aware Extension to Persistent Neighbor Alltoallv
// Needs global indices for each send and receive
int MPIX_Neighbor_locality_alltoallv_init(
        const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        const long global_sindices[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        const long global_rindices[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm,
        MPI_Info info,
        MPIX_Request** request_ptr);
int MPIX_Neighbor_part_locality_alltoallv_init(
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


void init_locality(const int n_sends, 
        const int* send_procs, 
        const int* send_indptr,
        const int n_recvs,
        const int* recv_procs,
        const int* recv_indptr,
        const long* global_send_indices,
        const long* global_recv_indices,
        const MPI_Datatype sendtype, 
        const MPI_Datatype recvtype,
        const MPIX_Comm* mpix_comm,
        MPIX_Request* request);


#ifdef __cplusplus
}
#endif

#endif
