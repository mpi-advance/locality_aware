#include "dist_topo.h"
#include <string.h>

int MPIX_Topo_dist_graph_create_adjacent(
        int indegree,
        const int sources[],
        const int sourceweights[],
        int outdegree,
        const int destinations[],
        const int destweights[],
        MPI_Info info,
        int reorder,
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

    // Create copy of sources/destination weights in MPIX_Topo struct
    if(sourceweights != MPI_UNWEIGHTED)
    {
        mpix_topo->sourceweights = (int *)malloc(indegree * sizeof(int));
        memcpy(mpix_topo->sourceweights, sourceweights, indegree * sizeof(int));
    }
    else
    {
        mpix_topo->sourceweights = MPI_UNWEIGHTED;
    }

    if(destweights != MPI_UNWEIGHTED)
    {
        mpix_topo->destweights = (int *)malloc(outdegree * sizeof(int));
        memcpy(mpix_topo->destweights, destweights, outdegree * sizeof(int));
    }
    else
    {
        mpix_topo->destweights = MPI_UNWEIGHTED;
    }

    *mpix_topo_ptr = mpix_topo;

    return MPI_SUCCESS;
}


int MPIX_Topo_free(MPIX_Topo* mpix_topo)
{
    free(mpix_topo->sources);
    if(mpix_topo->sourceweights != MPI_UNWEIGHTED)
        free(mpix_topo->sourceweights);
    free(mpix_topo->destinations);
    if(mpix_topo->destweights != MPI_UNWEIGHTED)
        free(mpix_topo->destweights);
    free(mpix_topo);
    return MPI_SUCCESS;
}

int MPIX_Topo_dist_graph_neighbors_count(MPIX_Topo* topo,
        int* indegree,
        int* outdegree,
        int* weighted)
{
    *indegree = topo->indegree;
    *outdegree = topo->outdegree;
    *weighted = (topo->sourceweights == MPI_UNWEIGHTED || topo->destweights == MPI_UNWEIGHTED);
    return MPI_SUCCESS;
}

int MPIX_Topo_dist_graph_neighbors(MPIX_Topo* topo,
        int maxindegree,
        int sources[],
        int sourceweights[],
        int maxoutdegree,
        int destinations[],
        int destweights[])
{
    memcpy(sources, topo->sources, maxindegree * sizeof(int));
    memcpy(destinations, topo->destinations, maxoutdegree * sizeof(int));

    if(topo->sourceweights != MPI_UNWEIGHTED)
        memcpy(sourceweights, topo->sourceweights, maxindegree * sizeof(int));

    if(topo->destweights != MPI_UNWEIGHTED)
        memcpy(destweights, topo->destweights, maxoutdegree * sizeof(int));
        
    return MPI_SUCCESS;
}
