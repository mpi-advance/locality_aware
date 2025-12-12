#ifndef MPI_ADVANCE_ALLGATHER_H
#define MPI_ADVANCE_ALLGATHER_H

#include <mpi.h>
#include <stdlib.h>

#include "communicator/MPIL_Comm.h"
#include "utils/MPIL_Alloc.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Function pointer to allreduce implemenation
 * @details 
 * Uses the parameters of standard MPI_Allgather API, except replacing MPI_Comm with MPIL_Comm
 * most of the behavior is derived from internal parameters in MPIL_Comm.
 * MPIL_API allgather switch statement targets one of these.  
 * @param [in] sendbuf buffer containing data gather
 * @param [in] sendcount int number of items to be sent at each step
 * @param [in] sendtype MPI_Datatype 
 * @param [out] recvbuf buffer to receive all messages
 * @param [in] recvcount int number of items to be received at each step
 * @param [in] recvtype MPI_Datatype 
 * @param [in] comm MPIL_Comm used for context
 **/
typedef int (*allgather_ftn)(
    const void*, int, MPI_Datatype, void*, int, MPI_Datatype, MPIL_Comm*);
typedef int (*allgather_helper_ftn)(
    const void*, int, MPI_Datatype, void*, int, MPI_Datatype, MPIL_Comm*,
    MPIL_Alloc_ftn, MPIL_Free_ftn);

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
 **/
int allgather_ring(const void* sendbuf,
                   int sendcount,
                   MPI_Datatype sendtype,
                   void* recvbuf,
                   int recvcount,
                   MPI_Datatype recvtype,
                   MPIL_Comm* comm);

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
 **/
int allgather_bruck(const void* sendbuf,
                   int sendcount,
                   MPI_Datatype sendtype,
                   void* recvbuf,
                   int recvcount,
                   MPI_Datatype recvtype,
                   MPIL_Comm* comm);


/** @brief Calls underlying PMPI_Allgather implementation **/
int allgather_pmpi(const void* sendbuf,
                   int sendcount,
                   MPI_Datatype sendtype,
                   void* recvbuf,
                   int recvcount,
                   MPI_Datatype recvtype,
                   MPIL_Comm* comm);


int allgather_ring_helper(const void* sendbuf,
                   int sendcount,
                   MPI_Datatype sendtype,
                   void* recvbuf,
                   int recvcount,
                   MPI_Datatype recvtype,
                   MPIL_Comm* comm,
                   MPIL_Alloc_ftn alloc_ftn,
                   MPIL_Free_ftn free_ftn);
int allgather_bruck_helper(const void* sendbuf,
                   int sendcount,
                   MPI_Datatype sendtype,
                   void* recvbuf,
                   int recvcount,
                   MPI_Datatype recvtype,
                   MPIL_Comm* comm,
                   MPIL_Alloc_ftn alloc_ftn,
                   MPIL_Free_ftn free_ftn);




/** @brief Helper functions
 **/



#ifdef __cplusplus
}
#endif

#endif

