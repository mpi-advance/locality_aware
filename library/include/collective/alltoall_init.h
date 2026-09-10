#ifndef MPI_ADVANCE_ALLTOALL_INIT_H
#define MPI_ADVANCE_ALLTOALL_INIT_H

#include <mpi.h>
#include <stdlib.h>

#include "communicator/MPIL_Comm.hpp"
#include "communicator/MPIL_Info.h"
#include "persistent/MPIL_Request.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Function pointer to alltoall implemenation
 * @details
 * Uses the parameters of standard MPI_Alltoall API, except replacing MPI_Comm with
 * MPIL_Comm most of the behavior is derived from internal parameters in MPIL_Comm.
 * MPIL_API alltoall switch statement targets one of these.
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcount int number of items in sendbuff
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcount int number of items expected in recvbuff
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @param [in] info MPIL_Info used for hints
 * @param [out] req_ptr MPIL_Request** for persistent request object
 **/
typedef int (*alltoall_init_ftn)(
    const void*, const int, MPI_Datatype, void*, const int, MPI_Datatype, MPIL_Comm*,
    MPIL_Info*, MPIL_Request**);



//** External Wrappers
//**//----------------------------------------------------------------------
/** @brief Call the pairwise implementation.
 * @details calls get_tag() then call pairwise_helper() with the same input parameters
 * plus the found tag.
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcount int number of items in sendbuff
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcount int number of items expected in recvbuff
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @param [in] info MPIL_Info used for hints
 * @param [out] req_ptr MPIL_Request** for persistent request object
 **/
int alltoall_init_pairwise(const void* sendbuf,
                      const int sendcount,
                      MPI_Datatype sendtype,
                      void* recvbuf,
                      const int recvcount,
                      MPI_Datatype recvtype,
                      MPIL_Comm* comm,
                      MPIL_Info* info,
                      MPIL_Request** req_ptr);
int alltoall_pairwise_start(MPIL_Request* request);
int alltoall_pairwise_wait(MPIL_Request* request, MPI_Status* status);

/** @brief Call the non-blocking implemenation.
 * @details calls get_tag then call nonblocking_helper() with the same input parameters
 * plus the found tag.
 *
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcount int number of items in sendbuff
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcount int number of items expected in recvbuff
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @param [in] info MPIL_Info used for hints
 * @param [out] req_ptr MPIL_Request** for persistent request object
 **/
int alltoall_init_nonblocking(const void* sendbuf,
                         const int sendcount,
                         MPI_Datatype sendtype,
                         void* recvbuf,
                         const int recvcount,
                         MPI_Datatype recvtype,
                         MPIL_Comm* comm,
                         MPIL_Info* info,
                         MPIL_Request** req_ptr);
int alltoall_nonblocking_start(MPIL_Request* request);
int alltoall_nonblocking_wait(MPIL_Request* request, MPI_Status* status);

/** @brief Call the RMA implemenation.
 * @details calls get_tag then call nonblocking_helper() with the same input parameters
 * plus the found tag.
 *
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcount int number of items in sendbuff
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcount int number of items expected in recvbuff
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @param [in] info MPIL_Info used for hints
 * @param [out] req_ptr MPIL_Request** for persistent request object
 **/
int alltoall_init_rma(const void* sendbuf,
                      const int sendcount,
                      MPI_Datatype sendtype,
                      void* recvbuf,
                      const int recvcount,
                      MPI_Datatype recvtype,
                      MPIL_Comm* comm,
                      MPIL_Info* info,
                      MPIL_Request** req_ptr);
int alltoall_rma_start(MPIL_Request* request);
int alltoall_rma_wait(MPIL_Request* request, MPI_Status* status);



#if defined(MPI4)
/** @brief calls underlying PMPI_Alltoall implementation **/
int alltoall_init_pmpi(const void* sendbuf,
                  const int sendcount,
                  MPI_Datatype sendtype,
                  void* recvbuf,
                  const int recvcount,
                  MPI_Datatype recvtype,
                  MPIL_Comm* comm,
                  MPIL_Info* info,
                  MPIL_Request** req_ptr);
#endif

#ifdef __cplusplus
}
#endif

#endif
