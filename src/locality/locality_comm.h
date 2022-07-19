#ifndef MPI_ADVANCE_LOCALITY_COMM_H
#define MPI_ADVANCE_LOCALITY_COMM_H

#include "comm_pkg.h"
#include "topology.h"

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
    
    const MPIX_Comm* communicators;
} LocalityComm;

void init_locality_comm(LocalityComm** locality_ptr, const MPIX_Comm* comm,
        MPI_Datatype sendtype, MPI_Datatype recvtype);
void finalize_locality_comm(LocalityComm* locality);
void destroy_locality_comm(LocalityComm* locality);

int MPIX_Comm_init(MPIX_Comm** comm_dist_graph_ptr, MPI_Comm global_comm);
int MPIX_Comm_free(MPIX_Comm* comm_dist_graph);


// Declarations of C++ methods
#ifdef __cplusplus
}
#endif

#endif
