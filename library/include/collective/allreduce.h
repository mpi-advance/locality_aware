#ifndef MPI_ADVANCE_ALLREDUCE_H
#define MPI_ADVANCE_ALLREDUCE_H

#include <mpi.h>
#include <stdlib.h>

#include "communicator/MPIL_Comm.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Function pointer to allreduce implemenation
 * @details 
 * Uses the parameters of standard MPI_Allreduce API, except replacing MPI_Comm with MPIL_Comm
 * most of the behavior is derived from internal parameters in MPIL_Comm.
 * MPIL_API allreduce switch statement targets one of these.  
 * @param [in] sendbuf buffer containing data to reduce
 * @param [out] recvbuf buffer to receive and reduce all messages
 * @param [in] count int number of items to be reduced
 * @param [in] datatype MPI_Datatype 
 * @param [in] op MPI_Op
 * @param [in] comm MPIL_Comm used for context
 **/
typedef int (*allreduce_ftn)(
    const void*, void*, const int, MPI_Datatype, MPI_Op op, MPIL_Comm*);

//** External Wrappers
//**//----------------------------------------------------------------------
/** @brief Call the recursive doubling implementation
 * @details If not power of 2 process count, will switch to dissemination
 * Otherwise, will perform the standard recursive doubling algorithm.
 * At each step i, all ranks exchange data with ranks 2^i processes away
 * @param [in] sendbuf buffer containing data to reduce
 * @param [out] recvbuf buffer to receive and reduce all messages
 * @param [in] count int number of items to be reduced
 * @param [in] datatype MPI_Datatype 
 * @param [in] op MPI_Op
 * @param [in] comm MPIL_Comm used for context
 **/
int allreduce_recursive_doubling(const void* sendbuf,
                                 void* recvbuf,
                                 int count,
                                 MPI_Datatype datatype,
                                 MPI_Op op,
                                 MPIL_Comm* comm);

/** @brief Call the locality-aware dissemination implementation
 * @details Each step consists of a local allreduce before non-local
 * communication.  Each process per node uses a separate stride.
 * Each local rank i initially sends to node - i - 1, and at each
 * successive step, this stride is multiplied by ppn.
 * @param [in] sendbuf buffer containing data to reduce
 * @param [out] recvbuf buffer to receive and reduce all messages
 * @param [in] count int number of items to be reduced
 * @param [in] datatype MPI_Datatype 
 * @param [in] op MPI_Op
 * @param [in] comm MPIL_Comm used for context
 **/
int allreduce_dissemination_loc(const void* sendbuf,
                                 void* recvbuf,
                                 int count,
                                 MPI_Datatype datatype,
                                 MPI_Op op,
                                 MPIL_Comm* comm);

/** @brief Call the locality-aware multi-leader dissemination implementation
 * @details Each step consists of a local allreduce before non-local
 * communication.  Each process per NUMA (4 per node) uses a separate stride.
 * Each local rank i initially sends to NUMA - i - 1, and at each
 * successive step, this stride is multiplied by ppNUMA.
 * @param [in] sendbuf buffer containing data to reduce
 * @param [out] recvbuf buffer to receive and reduce all messages
 * @param [in] count int number of items to be reduced
 * @param [in] datatype MPI_Datatype
 * @param [in] op MPI_Op
 * @param [in] comm MPIL_Comm used for context
 **/
int allreduce_dissemination_ml(const void* sendbuf,
                                 void* recvbuf,
                                 int count,
                                 MPI_Datatype datatype,
                                 MPI_Op op,
                                 MPIL_Comm* comm);

/** @brief Calls underlying PMPI_Allreduce implementation **/
int allreduce_pmpi(const void* sendbuf,
                                 void* recvbuf,
                                 int count,
                                 MPI_Datatype datatype,
                                 MPI_Op op,
                                 MPIL_Comm* comm);




#ifdef __cplusplus
}
#endif

#endif

