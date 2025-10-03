#ifndef MPI_ADVANCE_DIST_TOPO_H
#define MPI_ADVANCE_DIST_TOPO_H

#include "../communicator/locality_comm.h"
#include "../communicator/mpil_comm.h"
#include "mpi.h"

// Declarations of C++ methods
#ifdef __cplusplus
extern "C" {
#endif

typedef struct _MPIL_Topo
{
    int indegree;
    int* sources;
    int* sourceweights;
    int outdegree;
    int* destinations;
    int* destweights;
    int reorder;
} MPIL_Topo;

int MPIL_Topo_init(int indegree,
                   const int sources[],
                   const int sourceweights[],
                   int outdegree,
                   const int destinations[],
                   const int destweights[],
                   MPIL_Info* info,
                   MPIL_Topo** mpix_topo_ptr);

int MPIL_Topo_from_neighbor_comm(MPIL_Comm* comm, MPIL_Topo** mpix_topo_ptr);

int MPIL_Topo_free(MPIL_Topo** topo);

#ifdef __cplusplus
}
#endif

#endif
