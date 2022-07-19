#ifndef MPI_ADVANCE_LOCALITY_COMM_H
#define MPI_ADVANCE_LOCALITY_COMM_H

#include "comm_pkg.h"
#include <mpi.h>

// Declarations of C++ methods
#ifdef __cplusplus
extern "C"
{
#endif
    
typedef struct _MPIX_Comm
{
    MPI_Comm global_comm;
    MPI_Comm local_comm;
    MPI_Comm neighbor_comm;

    int num_nodes;
    int rank_node;
    int ppn;
} MPIX_Comm;

typedef struct _LocalityComm
{
    CommPkg* local_L_comm;
    CommPkg* local_S_comm;
    CommPkg* local_R_comm;
    CommPkg* global_comm;
    
    const MPIX_Comm* communicators;
} LocalityComm;

typedef struct _MPIX_Request
{
    int local_L_n_msgs;
    int local_S_n_msgs;
    int local_R_n_msgs;
    int global_n_msgs;

    MPI_Request* local_L_requests;
    MPI_Request* local_S_requests;
    MPI_Request* local_R_requests;
    MPI_Request* global_requests;

    LocalityComm* locality;
} MPIX_Request;

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
