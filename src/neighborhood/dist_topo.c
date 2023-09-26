#include "dist_topo.h"

int MPIX_Topo_dist_graph_adjacent(MPI_Comm comm, 
        int indegree,
        const int sources[],
        int outdegree,
        const int destinations[],
        MPI_Info info,
        MPIX_Topo** mpix_topo_ptr)
{
    MPIX_Topo* mpix_topo = (MPIX_Topo*)malloc(sizeof(MPIX_Topo));

    // Copy indegree and outdegree into MPIX_Topo struct
    // Create copy of sources/destinations in MPIX_Topo struct

    *mpix_topo_ptr = mpix_topo;
}


int MPIX_Topo_free(MPIX_Topo* mpix_topo)
{
    // Free sources and destinations in struct
    
    free(mpix_topo);
}

int MPIX_Topo_dist_graph_neighbors_count(MPIX_Topo* topo,
        int* indegree,
        int* outdegree)
{
    *indegree = topo->indegree;
    *outdegree = topo->outdegree;
}

int MPIX_Topo_dist_graph_neighbors(MPIX_Topo* topo,
        int indegree,
        int outdegree,
        int* sources,
        int* destinations)
{
    memcpy(sources, topo->sources, indegree);
    memcpy(destinations, topo->destinations, outdegree);
}
