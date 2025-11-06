#ifndef MPI_ADVANCE_LOCALITY_COMM_H
#define MPI_ADVANCE_LOCALITY_COMM_H

#include <mpi.h>

#include "MPIL_Comm.h"
#include "comm_pkg.h"

// Declarations of C++ methods
#ifdef __cplusplus
extern "C" {
#endif
/** @brief struct for messaging metadata **/
typedef struct LocalityComm
{
	/** @brief metadata on local (on-node) communications**/
    CommPkg* local_L_comm; 
    /** @brief metadata on original messages before aggration**/
	CommPkg* local_S_comm;  
    /** @brief metadata to redistribute aggravated messages.**/
	CommPkg* local_R_comm; 
    /** @brief metadata on messages sent between nodes.**/
	CommPkg* global_comm;   
	/** @brief array of communicators used in the locality mapping **/
    MPIL_Comm* communicators;
} LocalityComm;

void init_locality_comm(LocalityComm** locality_ptr,
                        MPIL_Comm* comm,
                        MPI_Datatype sendtype,
                        MPI_Datatype recvtype);
void finalize_locality_comm(LocalityComm* locality);
void destroy_locality_comm(LocalityComm* locality);
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
