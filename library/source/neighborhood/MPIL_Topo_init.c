#include "locality_aware.h"
#include "neighborhood/MPIL_Topo.h"

#include <cstdlib>
#include <cstring>

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
