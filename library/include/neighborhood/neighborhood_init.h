#ifndef MPI_ADVANCE_NEIGHBOR_INIT2_H
#define MPI_ADVANCE_NEIGHBOR_INIT2_H

#include "MPIL_Topo.h"
#include "communicator/MPIL_Comm.h"
#include "communicator/MPIL_Info.h"
#include "persistent/MPIL_Request.h"

// Starting locality-aware requests
// 1. Start Local_L
// 2. Start and wait for local_S
// 3. Start global
int neighbor_start(MPIL_Request* request);

// Wait for locality-aware requests
// 1. Wait for global
// 2. Start and wait for local_R
// 3. Wait for local_L
int neighbor_wait(MPIL_Request* request, MPI_Status* status);

void init_neighbor_request(MPIL_Request** request_ptr);

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

int neighbor_alltoallv_init_standard(const void* sendbuf,
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

int neighbor_alltoallv_init_locality(const void* sendbuf,
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

int neighbor_alltoallv_init_locality_ext(const void* sendbuffer,
                                         const int sendcounts[],
                                         const int sdispls[],
                                         const long global_sindices[],
                                         MPI_Datatype sendtype,
                                         void* recvbuffer,
                                         const int recvcounts[],
                                         const int rdispls[],
                                         const long global_rindices[],
                                         MPI_Datatype recvtype,
                                         MPIL_Topo* topo,
                                         MPIL_Comm* comm,
                                         MPIL_Info* info,
                                         MPIL_Request** request_ptr);

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
                   MPIL_Comm* mpil_comm,
                   MPIL_Request* request);

int init_communication(const void* sendbuffer,
                       int n_sends,
                       const int* send_procs,
                       const int* send_ptr,
                       MPI_Datatype sendtype,
                       void* recvbuffer,
                       int n_recvs,
                       const int* recv_procs,
                       const int* recv_ptr,
                       MPI_Datatype recvtype,
                       int tag,
                       MPI_Comm comm,
                       int* n_request_ptr,
                       MPI_Request** request_ptr);

#endif
