#ifndef MPI_ADVANCE_ALLREDUCE_INIT_H
#define MPI_ADVANCE_ALLREDUCE_INIT_H

#include <mpi.h>
#include <stdlib.h>

#include "communicator/MPIL_Comm.h"
#include "communicator/MPIL_Info.h"
#include "persistent/MPIL_Request.h"
#include "utils/MPIL_Alloc.h"

#ifdef __cplusplus
extern "C" {
#endif

int allreduce_recursive_doubling_start(MPIL_Request* request);
int allreduce_recursive_doubling_wait(MPIL_Request* request, MPI_Status* status);
int allreduce_dissemination_loc_start(MPIL_Request* request);
int allreduce_dissemination_loc_wait(MPIL_Request* request, MPI_Status* status);
int allreduce_dissemination_ml_start(MPIL_Request* request);
int allreduce_dissemination_ml_wait(MPIL_Request* request, MPI_Status* status);

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
typedef int (*allreduce_init_ftn)(
    const void*, void*, int, MPI_Datatype, MPI_Op op, MPIL_Comm*,
    MPIL_Info* info, MPIL_Request** req_ptr);
typedef int (*allreduce_init_helper_ftn)(
    const void*, void*, int, MPI_Datatype, MPI_Op op, MPIL_Comm*,
    MPIL_Info* info, MPIL_Request** req_ptr,
    MPIL_Alloc_ftn, MPIL_Free_ftn);

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
 * @param [in] info MPIL_Info used for hints
 * @param [out] req_ptr MPIL_Request** returns pointer to persistent request
 **/
int allreduce_recursive_doubling_init(const void* sendbuf,
                                 void* recvbuf,
                                 int count,
                                 MPI_Datatype datatype,
                                 MPI_Op op,
                                 MPIL_Comm* comm,
                                 MPIL_Info* info,
                                 MPIL_Request** req_ptr);

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
 * @param [in] info MPIL_Info used for hints
 * @param [out] req_ptr MPIL_Request** returns pointer to persistent request
 **/
int allreduce_dissemination_loc_init(const void* sendbuf,
                                 void* recvbuf,
                                 int count,
                                 MPI_Datatype datatype,
                                 MPI_Op op,
                                 MPIL_Comm* comm,                                 
                                 MPIL_Info* info,
                                 MPIL_Request** req_ptr);


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
 * @param [in] info MPIL_Info used for hints
 * @param [out] req_ptr MPIL_Request** returns pointer to persistent request
 **/
int allreduce_dissemination_ml_init(const void* sendbuf,
                                 void* recvbuf,
                                 int count,
                                 MPI_Datatype datatype,
                                 MPI_Op op,
                                 MPIL_Comm* comm,
                                 MPIL_Info* info,
                                 MPIL_Request** req_ptr);


/** @brief Calls underlying PMPI_Allreduce implementation **/
int allreduce_pmpi_init(const void* sendbuf,
                                 void* recvbuf,
                                 int count,
                                 MPI_Datatype datatype,
                                 MPI_Op op,
                                 MPIL_Comm* comm,
                                 MPIL_Info* info,
                                 MPIL_Request** req_ptr);



/** @brief Helper functions
 * @details takes the temporary buffer as input
 * @details Each step consists of a local allreduce before non-local
 **/
int allreduce_recursive_doubling_init_helper(
                                 const void* sendbuf,
                                 void* recvbuf,
                                 int count,
                                 MPI_Datatype datatype,
                                 MPI_Op op,
                                 MPIL_Comm* comm,
                                 MPIL_Info* info,
                                 MPIL_Request** req_ptr,
                                 MPIL_Alloc_ftn alloc_ftn,
                                 MPIL_Free_ftn free_ftn);

int allreduce_dissemination_loc_init_helper(
                                 const void* sendbuf,
                                 void* recvbuf,
                                 int count,
                                 MPI_Datatype datatype,
                                 MPI_Op op,
                                 MPIL_Comm* comm,
                                 MPIL_Info* info,
                                 MPIL_Request** req_ptr,
                                 MPIL_Alloc_ftn alloc_ftn,
                                 MPIL_Free_ftn free_ftn);

int allreduce_dissemination_ml_init_helper(
                                 const void* sendbuf,
                                 void* recvbuf,
                                 int count,
                                 MPI_Datatype datatype,
                                 MPI_Op op,
                                 MPIL_Comm* comm,
                                 MPIL_Info* info,
                                 MPIL_Request** req_ptr,
                                 MPIL_Alloc_ftn alloc_ftn,
                                 MPIL_Free_ftn free_ftn);

int allreduce_dissemination_loc_init_core(
                                 const void* sendbuf,
                                 void* recvbuf,
                                 int count,
                                 MPI_Datatype datatype,
                                 MPI_Op op,
                                 MPI_Comm global_comm,
                                 MPI_Comm group_comm,
                                 MPI_Comm local_comm,
                                 MPIL_Info* info,
                                 int tag,
                                 MPIL_Request** req_ptr,
                                 MPIL_Alloc_ftn alloc_ftn,
                                 MPIL_Free_ftn free_ftn);


#ifdef __cplusplus
}
#endif

#endif

