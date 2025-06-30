#ifndef MPI_ADVANCE_DIST_GRAPH_H
#define MPI_ADVANCE_DIST_GRAPH_H

#include "mpi.h"
#include "communicator/locality_comm.h"
#include "communicator/mpix_comm.h"

// Declarations of C++ methods
#ifdef __cplusplus
extern "C"
{
#endif

int MPIX_Dist_graph_create_adjacent(MPI_Comm comm_old, 
        int indegree,
        const int sources[],
        const int sourceweights[],
        int outdegree,
        const int destinations[],
        const int destweights[],
        MPIX_Info* info,
        int reorder,
        MPIX_Comm** comm_dist_graph_ptr);

#ifdef __cplusplus
}
#endif


#endif
