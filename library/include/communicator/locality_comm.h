#ifndef MPI_ADVANCE_LOCALITY_COMM_H
#define MPI_ADVANCE_LOCALITY_COMM_H

#include <mpi.h>

#include "comm_pkg.h"
#include "locality_aware.h"

// Declarations of C++ methods
#ifdef __cplusplus
extern "C" {
#endif

//typedef struct _CommPkg CommPkg; 

typedef struct LocalityComm
{
    CommPkg* local_L_comm;  /**< ??? **/
    CommPkg* local_S_comm;  /**< ??? **/
    CommPkg* local_R_comm;  /**< ??? **/
    CommPkg* global_comm;   /**< ??? **/

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
