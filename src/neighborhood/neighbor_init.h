#ifndef MPI_ADVANCE_NEIGHBOR_PERSISTENT_H
#define MPI_ADVANCE_NEIGHBOR_PERSISTENT_H

#include "communicator/locality_comm.h"
#include "persistent/persistent.h"
#include "neighbor.h"

#ifdef __cplusplus
extern "C"
{
#endif

enum NeighborAlltoallvInitMethod { NEIGHBOR_ALLTOALLV_INIT_STANDARD, NEIGHBOR_ALLTOALLV_INIT_LOCALITY};
extern NeighborAlltoallvInitMethod mpix_neighbor_alltoallv_init_implementation;

typedef int (*neighbor_alltoallv_init_ftn)(
        const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Topo* topo,
        MPIX_Comm* comm,
        MPIX_Info* info,
        MPIX_Request** request_ptr);

// Starting locality-aware requests
// 1. Start Local_L
// 2. Start and wait for local_S
// 3. Start global
int neighbor_start(MPIX_Request* request);


// Wait for locality-aware requests
// 1. Wait for global
// 2. Start and wait for local_R
// 3. Wait for local_L
int neighbor_wait(MPIX_Request* request, MPI_Status* status);


void init_neighbor_request(MPIX_Request** request_ptr);


// Standard Persistent Neighbor Alltoallv
// Extension takes array of requests instead of single request
// 'requests' must be of size indegree+outdegree!
int MPIX_Neighbor_alltoallv_init_topo(
        const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Topo* topo,
        MPIX_Comm* comm,
        MPIX_Info* info,
        MPIX_Request** request_ptr);
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
        MPIX_Info* info,
        MPIX_Request** request_ptr);


// Locality-Aware Extension to Persistent Neighbor Alltoallv
// Needs global indices for each send and receive
int MPIX_Neighbor_alltoallv_init_ext_topo(
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
        MPIX_Topo* topo,
        MPIX_Comm* comm,
        MPIX_Info* info,
        MPIX_Request** request_ptr);
int MPIX_Neighbor_alltoallv_init_ext(
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
        MPIX_Info* info,
        MPIX_Request** request_ptr);




int neighbor_alltoallv_init_standard(
        const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Topo* topo,
        MPIX_Comm* comm,
        MPIX_Info* info,
        MPIX_Request** request_ptr);
int neighbor_alltoallv_init_locality(
        const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Topo* topo,
        MPIX_Comm* comm,
        MPIX_Info* info,
        MPIX_Request** request_ptr);
int neighbor_alltoallv_init_locality_ext(
        const void* sendbuffer,
        const int sendcounts[],
        const int sdispls[],
        const long global_sindices[],
        MPI_Datatype sendtype,
        void* recvbuffer,
        const int recvcounts[],
        const int rdispls[],
        const long global_rindices[],
        MPI_Datatype recvtype,
        MPIX_Topo* topo,
        MPIX_Comm* comm,
        MPIX_Info* info,
        MPIX_Request** request_ptr);





void init_locality(const int n_sends,
        const int* send_procs,
        const int* send_indptr,
        const int* sendcounts,
        const int n_recvs,
        const int* recv_procs,
        const int* recv_indptr,
        const int* recvcounts,
        const long* global_send_indices,
        const long* global_recv_indices,
        const MPI_Datatype sendtype,
        const MPI_Datatype recvtype,
        MPIX_Comm* mpix_comm,
        MPIX_Request* request);

#ifdef __cplusplus
}
#endif

#endif

