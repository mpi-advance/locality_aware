#include "../../../include/neighborhood/dist_topo.h"

#include <string.h>

int MPIL_Topo_init(int indegree,
                   const int sources[],
                   const int sourceweights[],
                   int outdegree,
                   const int destinations[],
                   const int destweights[],
                   MPIL_Info* info,
                   MPIL_Topo** mpix_topo_ptr)
{
    MPIL_Topo* mpix_topo = (MPIL_Topo*)malloc(sizeof(MPIL_Topo));

    // Copy indegree and outdegree into MPIL_Topo struct
    mpix_topo->indegree  = indegree;
    mpix_topo->outdegree = outdegree;

    // Create copy of sources/destinations in MPIL_Topo struct
    mpix_topo->sources      = NULL;
    mpix_topo->destinations = NULL;

    if (indegree)
    {
        mpix_topo->sources = (int*)malloc(indegree * sizeof(int));
        memcpy(mpix_topo->sources, sources, indegree * sizeof(int));
        if (sourceweights != MPI_UNWEIGHTED)
        {
            mpix_topo->sourceweights = (int*)malloc(indegree * sizeof(int));
            memcpy(mpix_topo->sourceweights, sourceweights, indegree * sizeof(int));
        }
        else
        {
            mpix_topo->sourceweights = MPI_UNWEIGHTED;
        }
    }

    if (outdegree)
    {
        mpix_topo->destinations = (int*)malloc(outdegree * sizeof(int));
        memcpy(mpix_topo->destinations, destinations, outdegree * sizeof(int));

        if (destweights != MPI_UNWEIGHTED)
        {
            mpix_topo->destweights = (int*)malloc(outdegree * sizeof(int));
            memcpy(mpix_topo->destweights, destweights, outdegree * sizeof(int));
        }
        else
        {
            mpix_topo->destweights = MPI_UNWEIGHTED;
        }
    }

    *mpix_topo_ptr = mpix_topo;

    return MPI_SUCCESS;
}

int MPIL_Topo_from_neighbor_comm(MPIL_Comm* comm, MPIL_Topo** mpix_topo_ptr)
{
    MPIL_Topo* mpix_topo = (MPIL_Topo*)malloc(sizeof(MPIL_Topo));

    int weighted;
    MPI_Dist_graph_neighbors_count(
        comm->neighbor_comm, &(mpix_topo->indegree), &(mpix_topo->outdegree), &weighted);

    // Create copy of sources/destinations in MPIL_Topo struct
    mpix_topo->sources      = NULL;
    mpix_topo->destinations = NULL;

    if (mpix_topo->indegree)
    {
        mpix_topo->sources = (int*)malloc(mpix_topo->indegree * sizeof(int));
        if (weighted)
        {
            mpix_topo->sourceweights = (int*)malloc(mpix_topo->indegree * sizeof(int));
        }
        else
        {
            mpix_topo->sourceweights = MPI_UNWEIGHTED;
        }
    }

    if (mpix_topo->outdegree)
    {
        mpix_topo->destinations = (int*)malloc(mpix_topo->outdegree * sizeof(int));
        if (weighted)
        {
            mpix_topo->destweights = (int*)malloc(mpix_topo->outdegree * sizeof(int));
        }
        else
        {
            mpix_topo->destweights = MPI_UNWEIGHTED;
        }
    }

    MPI_Dist_graph_neighbors(comm->neighbor_comm,
                             mpix_topo->indegree,
                             mpix_topo->sources,
                             mpix_topo->sourceweights,
                             mpix_topo->outdegree,
                             mpix_topo->destinations,
                             mpix_topo->destweights);

    *mpix_topo_ptr = mpix_topo;

    return MPI_SUCCESS;
}

int MPIL_Topo_free(MPIL_Topo** mpix_topo_ptr)
{
    MPIL_Topo* mpix_topo = *mpix_topo_ptr;

    if (mpix_topo->indegree)
    {
        free(mpix_topo->sources);
        if (mpix_topo->sourceweights != MPI_UNWEIGHTED)
        {
            free(mpix_topo->sourceweights);
        }
    }

    if (mpix_topo->outdegree)
    {
        free(mpix_topo->destinations);
        if (mpix_topo->destweights != MPI_UNWEIGHTED)
        {
            free(mpix_topo->destweights);
        }
    }
    free(mpix_topo);
    return MPI_SUCCESS;
}
