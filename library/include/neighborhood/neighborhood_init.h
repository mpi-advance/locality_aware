#ifndef MPI_ADVANCE_NEIGHBOR_INIT_H
#define MPI_ADVANCE_NEIGHBOR_INIT_H

#include "MPIL_Topo.h"
#include "communicator/MPIL_Comm.hpp"
#include "communicator/MPIL_Info.h"
#include "persistent/MPIL_Request.h"

#ifdef __cplusplus
extern "C" {
#endif

void init_neighbor_request(MPIL_Request** request_ptr);
int neighbor_start(MPIL_Request* request);
int neighbor_wait(MPIL_Request* request, MPI_Status* status);
int neighbor_a2a_start(MPIL_Request* request);
int neighbor_a2a_wait(MPIL_Request* request, MPI_Status* status);
int neighbor_rma_start(MPIL_Request* request);
int neighbor_rma_wait(MPIL_Request* request, MPI_Status* status);
int neighbor_pscw_start(MPIL_Request* request);
int neighbor_pscw_wait(MPIL_Request* request, MPI_Status* status);



/** @brief Function pointer to persistent neighbor alltoallv implemenation
 * @details
 * Uses the parameters of standard MPI_Neighbor_alltoallv API, 
 * except replacing MPI_Comm with
 * MPIL_Comm most of the behavior is derived from internal parameters in MPIL_Comm.
 * MPIL_API neighbor alltoallv switch statement targets one of these.
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcounts int* number of items per send
 * @param [in] sdispls int* displacement per send
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcounts int* number of items per recv
 * @param [in] rdispls int* displacement per recv
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @param [in] info MPIL_Info used for hints
 * @param [out] req_ptr MPIL_Request** for persistent request object
 **/
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

//** External Wrappers
//**//----------------------------------------------------------------------
/** @brief Call the standard send_init/recv_init implementation.
 * @details uses MPI_Send_init and MPI_Recv_init to perform
 * a single send and receive per required message
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcounts int* number of items per send
 * @param [in] sdispls int* displacement per send
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcounts int* number of items per recv
 * @param [in] rdispls int* displacement per recv
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @param [in] info MPIL_Info used for hints
 * @param [out] req_ptr MPIL_Request** for persistent request object
 **/
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

/** @brief Call the locality-aware aggregated implementation.
 * @details performs 2-step aggregation, sending a single message
 * to the corresponding local_rank on the receiving node,
 * and finally disaggregating messages on the receiving nodes.
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcounts int* number of items per send
 * @param [in] sdispls int* displacement per send
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcounts int* number of items per recv
 * @param [in] rdispls int* displacement per recv
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @param [in] info MPIL_Info used for hints
 * @param [out] req_ptr MPIL_Request** for persistent request object
 **/
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

/** @brief Implements a neighbor alltoallv with a global
 * alltoallv under the hood.  For relatively dense communication patterns. 
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcounts int* number of items per send
 * @param [in] sdispls int* displacement per send
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcounts int* number of items per recv
 * @param [in] rdispls int* displacement per recv
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @param [in] info MPIL_Info used for hints
 * @param [out] req_ptr MPIL_Request** for persistent request object
 **/
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

/** @brief Implements a neighbor alltoallv with RMA puts.
 * Uses a fence for synchronization. 
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcounts int* number of items per send
 * @param [in] sdispls int* displacement per send
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcounts int* number of items per recv
 * @param [in] rdispls int* displacement per recv
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @param [in] info MPIL_Info used for hints
 * @param [out] req_ptr MPIL_Request** for persistent request object
 **/
int neighbor_alltoallv_init_rma(const void* sendbuf,
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

/** @brief Implements a neighbor alltoallv with RMA puts.
 * Uses a post-start-complete-wait for synchronization,
 * synchronizing only among neighbors rather than globally. 
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcounts int* number of items per send
 * @param [in] sdispls int* displacement per send
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcounts int* number of items per recv
 * @param [in] rdispls int* displacement per recv
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @param [in] info MPIL_Info used for hints
 * @param [out] req_ptr MPIL_Request** for persistent request object
 **/
int neighbor_alltoallv_init_pscw(const void* sendbuf,
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

int neighbor_alltoallv_init_rma_helper(const void* sendbuf,
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


/** @brief Call the extended locality-aware aggregated implementation.
 * @details performs 3-step aggregation, first aggregating data
 * on the sending node before sending at most a single message
 * between any two sets of nodes.  Finally messages are disaggregated
 * on the receiving node.
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcounts int* number of items per send
 * @param [in] sdispls int* displacement per send
 * @param [in] global_sindices long* global indices being sent
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcounts int* number of items per recv
 * @param [in] rdispls int* displacement per recv
 * @param [in] global_rindices long* global indices being recvd
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @param [in] info MPIL_Info used for hints
 * @param [out] req_ptr MPIL_Request** for persistent request object
 **/
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


/** @brief Call the extended collective implementation of 
 * a neighborhood alltoallv.  All unique indices to be sent
 * are gathered among all processes with an Allgatherv.
 * This operation has memory constraints, and should only be 
 * used for smaller and denser matrices (e.g. coarse levels of AMG).
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcounts int* number of items per send
 * @param [in] sdispls int* displacement per send
 * @param [in] global_sindices long* global indices being sent
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcounts int* number of items per recv
 * @param [in] rdispls int* displacement per recv
 * @param [in] global_rindices long* global indices being recvd
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @param [in] info MPIL_Info used for hints
 * @param [out] req_ptr MPIL_Request** for persistent request object
 **/
int neighbor_alltoallv_init_coll_ag(const void* sendbuffer,
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
