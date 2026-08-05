#ifndef MPI_ADVANCE_NEIGHBOR_INIT_H
#define MPI_ADVANCE_NEIGHBOR_INIT_H

#include "MPIL_Topo.h"
#include "communicator/MPIL_Comm.hpp"
#include "communicator/MPIL_Info.h"
#include "persistent/MPIL_Request.h"

#ifdef __cplusplus
extern "C" {
#endif

int neighbor_start(MPIL_Request* request);
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

int neighbor_alltoallv_init_standard_helper(const void* sendbuf,
                                     const int sendcounts[],
                                     const int sdispls[],
                                     MPI_Datatype sendtype,
                                     void* recvbuf,
                                     const int recvcounts[],
                                     const int rdispls[],
                                     MPI_Datatype recvtype,
                                     MPIL_Topo* topo,
                                     MPI_Comm comm,
                                     MPIL_Info* info,
                                     int tag,
                                     MPIL_Request* request);

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

int neighbor_alltoallv_init_locality_helper(const void* sendbuf,
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
                                     MPIL_Request** request_ptr,
                                     MPIL_Alloc_ftn alloc_ftn,
                                     MPIL_Free_ftn free_ftn);


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

int neighbor_alltoallv_init_locality_ext_helper(const void* sendbuffer,
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
                                         MPIL_Request** request_ptr,
                                         MPIL_Alloc_ftn alloc_ftn,
                                         MPIL_Free_ftn free_ftn);

int neighbor_alltoallv_init_coll_a2a(const void* sendbuf,
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

int neighbor_alltoallv_init_coll_a2a_helper(const void* sendbuf,
                                     const int sendcounts[],
                                     const int sdispls[],
                                     MPI_Datatype sendtype,
                                     void* recvbuf,
                                     const int recvcounts[],
                                     const int rdispls[],
                                     MPI_Datatype recvtype,
                                     MPIL_Topo* topo,
                                     MPI_Comm comm,
                                     MPIL_Info* info,
                                     int tag,
                                     MPIL_Request* request);

int neighbor_alltoallv_init_coll_ag(const void* sendbuf,
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

int neighbor_alltoallv_init_coll_ag_helper(const void* sendbuf,
                                     const int sendcounts[],
                                     const int sdispls[],
                                     MPI_Datatype sendtype,
                                     void* recvbuf,
                                     const int recvcounts[],
                                     const int rdispls[],
                                     MPI_Datatype recvtype,
                                     MPIL_Topo* topo,
                                     MPI_Comm comm,
                                     MPIL_Info* info,
                                     int tag,
                                     MPIL_Request* request);


void init_locality(const int n_sends,
                   const int* send_procs,
                   const int* send_indptr,
                   const int* sendcounts,
                   const void* sendbuffer,
                   const int n_recvs,
                   const int* recv_procs,
                   const int* recv_indptr,
                   const int* recvcounts,
                   void* recvbuffer,
                   const long* global_send_indices,
                   const long* global_recv_indices,
                   const MPI_Datatype sendtype,
                   const MPI_Datatype recvtype,
                   MPIL_Comm* mpil_comm,
                   MPIL_Request* request,
                   MPIL_Alloc_ftn alloc_ftn);


void init_packing_buffers(MPIL_Request* request, 
                            int size_sends, 
                            int* send_indices, 
                            int send_size, 
                            const void* _sendbuf, 
                            int size_recvs, 
                            int* recv_indices, 
                            int recv_size, 
                            void* _recvbuf,
                            MPIL_Alloc_ftn alloc_ftn);


#ifdef __cplusplus
}
#endif

#endif
