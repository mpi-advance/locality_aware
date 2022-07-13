#ifndef MPI_ADVANCE_DIST_GRAPH_H
#define MPI_ADVANCE_DIST_GRAPH_H

#include "mpi.h"
#include "topology.h"
#include "locality_comm.h"

typedef struct _MPIX_Comm
{
    MPI_Comm global_comm;
    MPI_Comm local_comm;
    MPI_Comm neighbor_comm;

    int num_nodes;
    int rank_node;
    int ppn;
} MPIX_Comm;

int MPIX_Dist_graph_create_adjacent(MPI_Comm comm_old, 
        int indegree,
        const int sources[],
        const int sourceweights[],
        int outdegree,
        const int destinations[],
        const int destweights[],
        MPI_Info info,
        int reorder,
        MPIX_Comm** comm_dist_graph_ptr);

int MPIX_Comm_free(MPIX_Comm* comm_dist_graph);

#endif
