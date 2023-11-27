#include "dist_topo.h"
#include <string.h>

int MPIX_Topo_dist_graph_adjacent(MPIX_Comm *comm, 
        int indegree,
        const int sources[],
        int outdegree,
        const int destinations[],
        MPI_Info info,
        MPIX_Topo** mpix_topo_ptr)
{
    MPIX_Topo* mpix_topo = (MPIX_Topo*)malloc(sizeof(MPIX_Topo));

    // Copy indegree and outdegree into MPIX_Topo struct
    mpix_topo->indegree = indegree;
    mpix_topo->outdegree = outdegree;

    // Create copy of sources/destinations in MPIX_Topo struct
    mpix_topo->sources = (int *)malloc(indegree * sizeof(int));
    mpix_topo->destinations = (int *)malloc(outdegree * sizeof(int));

    memcpy(mpix_topo->sources, sources, indegree * sizeof(int));
    memcpy(mpix_topo->destinations, destinations, outdegree * sizeof(int));

    *mpix_topo_ptr = mpix_topo;
}


int MPIX_Topo_free(MPIX_Topo* mpix_topo)
{
    free(mpix_topo->sources);
    free(mpix_topo->destinations);
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
    memcpy(sources, topo->sources, indegree * sizeof(int));
    memcpy(destinations, topo->destinations, outdegree * sizeof(int));
}
