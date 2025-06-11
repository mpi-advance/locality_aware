#ifndef MPI_ADVANCE_DIST_TOPO_H
#define MPI_ADVANCE_DIST_TOPO_H

#include "mpi.h"
#include "communicator/locality_comm.h"
#include "communicator/mpix_comm.h"

// Declarations of C++ methods
#ifdef __cplusplus
extern "C"
{
#endif
 

typedef struct _MPIX_Topo
{
    int indegree;
    int* sources;
    int* sourceweights;
    int outdegree;
    int* destinations;
    int* destweights;
    int reorder;
} MPIX_Topo;

int MPIX_Topo_init( 
        int indegree,
        const int sources[],
        const int sourceweights[],
        int outdegree,
        const int destinations[],
        const int destweights[],
        MPIX_Info* info,
        MPIX_Topo** mpix_topo_ptr);

int MPIX_Topo_from_neighbor_comm(
        MPIX_Comm* comm,
        MPIX_Topo** mpix_topo_ptr);

int MPIX_Topo_free(MPIX_Topo** topo);


#ifdef __cplusplus
}
#endif


#endif
