#ifndef MPI_ADVANCE_DIST_TOPO_H
#define MPI_ADVANCE_DIST_TOPO_H

#include "mpi.h"
#include "locality/locality_comm.h"
#include "locality/topology.h"

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

int MPIX_Topo_dist_graph_create_adjacent( 
        int indegree,
        const int sources[],
        const int sourceweights[],
        int outdegree,
        const int destinations[],
        const int destweights[],
        MPI_Info info,
        int reorder,
        MPIX_Topo** mpix_topo_ptr);

int MPIX_Topo_free(MPIX_Topo* topo);


int MPIX_Topo_dist_graph_neighbors_count(MPIX_Topo* topo,
        int* indegree,
        int* outdegree,
        int* weighted);

int MPIX_Topo_dist_graph_neighbors(MPIX_Topo* topo,
        int maxindegree,
        int sources[],
        int sourceweights[],
        int maxoutdegree,
        int desitnations[],
        int destweights[]);

#ifdef __cplusplus
}
#endif


#endif
