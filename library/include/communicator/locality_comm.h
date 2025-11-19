#ifndef MPI_ADVANCE_LOCALITY_COMM_H
#define MPI_ADVANCE_LOCALITY_COMM_H

#include <mpi.h>

#include "MPIL_Comm.h"
#include "comm_pkg.h"

// Declarations of C++ methods
#ifdef __cplusplus
extern "C" {
#endif
/** @brief Struct for messaging metadata for the different groups of processes**/
typedef struct LocalityComm
{
    /** @brief Metadata about local (on-node) communications. **/
    CommPkg* local_L_comm;
    /** @brief Metadata about original messages before aggregation. **/
    CommPkg* local_S_comm;
    /** @brief Metadata to redistribute aggravated messages. **/
    CommPkg* local_R_comm;
    /** @brief Metadata on messages sent between nodes. **/
    CommPkg* global_comm;
    /** @brief Pointer to MPIL_Comm used in the locality mapping. **/
    MPIL_Comm* communicators;
} LocalityComm;

/** @brief Constructor for ::LocalityComm. Datatypes used to create message metadata. */
void init_locality_comm(LocalityComm** locality_ptr,
                        MPIL_Comm* comm,
                        MPI_Datatype sendtype,
                        MPI_Datatype recvtype);
/** @brief  Finalize the ::CommPkg objects inside this object. */
void finalize_locality_comm(LocalityComm* locality);
/** @brief Destructor for a ::LocalityComm object. */
void destroy_locality_comm(LocalityComm* locality);
/** @brief Collect the maximum number of local and non-local messages.
 *  @details This method is currently collective over MPI_COMM_WORLD, using
 * MPI_Allreduce(MPI_MAX). For all ranks in that communicator, the first two values are
 * derived from LocalityComm::local_L_comm, LocalityComm::local_S_comm, and
 * LocalityComm::local_R_comm. The last two outputs are derived from
 * LocalityComm::global_comm.
 */
void get_local_comm_data(LocalityComm* locality,
                         int* max_local_num,
                         int* max_local_size,
                         int* max_non_local_num,
                         int* max_non_local_size);

// Declarations of C++ methods
#ifdef __cplusplus
}
#endif

#endif
