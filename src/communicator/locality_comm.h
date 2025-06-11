#ifndef MPI_ADVANCE_LOCALITY_COMM_H
#define MPI_ADVANCE_LOCALITY_COMM_H

#include "comm_pkg.h"
#include "mpix_comm.h"

#include <mpi.h>

// Declarations of C++ methods
#ifdef __cplusplus
extern "C"
{
#endif
    

typedef struct _LocalityComm
{
    CommPkg* local_L_comm;
    CommPkg* local_S_comm;
    CommPkg* local_R_comm;
    CommPkg* global_comm;
    
    MPIX_Comm* communicators;
} LocalityComm;

void init_locality_comm(LocalityComm** locality_ptr, MPIX_Comm* comm,
        MPI_Datatype sendtype, MPI_Datatype recvtype);
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
