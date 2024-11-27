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
    int outdegree;
    int* destinations;

} MPIX_Topo;

int MPIX_Topo_dist_graph_adjacent(MPIX_Comm *comm, 
        int indegree,
        const int sources[],
        int outdegree,
        const int destinations[],
        MPI_Info info,
        MPIX_Topo** mpix_topo_ptr);

int MPIX_Topo_free(MPIX_Topo* topo);


int MPIX_Topo_dist_graph_neighbors_count(MPIX_Topo* topo,
        int* indegree,
        int* outdegree);

int MPIX_Topo_dist_graph_neighbors(MPIX_Topo* topo,
        int indegree,
        int outdegree,
        int *sources,
        int *desitnations);

#ifdef __cplusplus
}
#endif


#endif
