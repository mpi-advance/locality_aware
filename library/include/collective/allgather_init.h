#ifndef MPI_ADVANCE_ALLGATHER_INIT_H
#define MPI_ADVANCE_ALLGATHER_INIT_H

#include <mpi.h>
#include <stdlib.h>

#include "communicator/MPIL_Comm.hpp"
#include "communicator/MPIL_Info.h"
#include "persistent/MPIL_Request.h"

#ifdef __cplusplus
extern "C" {
#endif


/** @brief Function pointer to persistent allreduce implemenation
 * @details 
 * Uses the parameters of standard MPI_Allgather_init API, except replacing MPI_Comm with MPIL_Comm
 * most of the behavior is derived from internal parameters in MPIL_Comm.
 * MPIL_API allgather init switch statement targets one of these.  
 * @param [in] sendbuf buffer containing data gather
 * @param [in] sendcount int number of items to be sent at each step
 * @param [in] sendtype MPI_Datatype 
 * @param [out] recvbuf buffer to receive all messages
 * @param [in] recvcount int number of items to be received at each step
 * @param [in] recvtype MPI_Datatype 
 * @param [in] comm MPIL_Comm used for context
 * @param [in] info MPIL_Info used for hints
 * @param [out] req_ptr MPIL_Request** pointer to persistent request object
 **/
typedef int (*allgather_init_ftn)(
    const void*, int, MPI_Datatype, void*, int, MPI_Datatype, MPIL_Comm*, 
    MPIL_Info*, MPIL_Request** req_ptr);

//** External Wrappers
//**//----------------------------------------------------------------------
/** @brief Call the ring implementation
 * @details Each process sends num_procs-1 messages to neighboring processes.
 * At each step, process p sends to p+1 and receives from p-1.
 * @param [in] sendbuf buffer containing data gather
 * @param [in] sendcount int number of items to be sent at each step
 * @param [in] sendtype MPI_Datatype 
 * @param [out] recvbuf buffer to receive all messages
 * @param [in] recvcount int number of items to be received at each step
 * @param [in] recvtype MPI_Datatype 
 * @param [in] comm MPIL_Comm used for context
 * @param [in] info MPIL_Info used for hints
 * @param [out] req_ptr MPIL_Request** pointer to persistent request object
 **/
int allgather_init_ring(const void* sendbuf,
                   int sendcount,
                   MPI_Datatype sendtype,
                   void* recvbuf,
                   int recvcount,
                   MPI_Datatype recvtype,
                   MPIL_Comm* comm,
                   MPIL_Info* info,
                   MPIL_Request** req_ptr);
int allgather_ring_start(MPIL_Request* request);
int allgather_ring_wait(MPIL_Request* request, MPI_Status* status);

//**//----------------------------------------------------------------------
/** @brief Call the Bruck implementation
 * @details Each process sends log(p) messages.
 * @param [in] sendbuf buffer containing data gather
 * @param [in] sendcount int number of items to be sent at each step
 * @param [in] sendtype MPI_Datatype 
 * @param [out] recvbuf buffer to receive all messages
 * @param [in] recvcount int number of items to be received at each step
 * @param [in] recvtype MPI_Datatype 
 * @param [in] comm MPIL_Comm used for context
 * @param [in] info MPIL_Info used for hints
 * @param [out] req_ptr MPIL_Request** pointer to persistent request object
 **/
int allgather_init_bruck(const void* sendbuf,
                   int sendcount,
                   MPI_Datatype sendtype,
                   void* recvbuf,
                   int recvcount,
                   MPI_Datatype recvtype,
                   MPIL_Comm* comm,
                   MPIL_Info* info,
                   MPIL_Request** req_ptr);
int allgather_bruck_start(MPIL_Request* request);
int allgather_bruck_wait(MPIL_Request* request, MPI_Status* status);


/** @brief Calls underlying PMPI_Allgather implementation **/

#if defined(MPI4)
int allgather_init_pmpi(const void* sendbuf,
                   int sendcount,
                   MPI_Datatype sendtype,
                   void* recvbuf,
                   int recvcount,
                   MPI_Datatype recvtype,
                   MPIL_Comm* comm,
                   MPIL_Info* info,
                   MPIL_Request** req_ptr);
#endif




#ifdef __cplusplus
}
#endif

#endif
