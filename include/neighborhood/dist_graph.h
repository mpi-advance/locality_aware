#ifndef MPI_ADVANCE_DIST_GRAPH_H
#define MPI_ADVANCE_DIST_GRAPH_H

//#include "../communicator/locality_comm.h"
#include "../communicator/mpil_comm.h"
#include "../communicator/MPIL_Info.h"
#include "mpi.h"

// Declarations of C++ methods
#ifdef __cplusplus
extern "C" {
#endif

int MPIL_Dist_graph_create_adjacent(MPI_Comm comm_old,
                                    int indegree,
                                    const int sources[],
                                    const int sourceweights[],
                                    int outdegree,
                                    const int destinations[],
                                    const int destweights[],
                                    MPIL_Info* info,
                                    int reorder,
                                    MPIL_Comm** comm_dist_graph_ptr);

#ifdef __cplusplus
}
#endif

#endif
