#include "locality_aware.h"
#include "neighborhood/MPIL_Topo.h"
#include <stdlib.h>
#include <string.h>

int MPIL_Topo_init(int indegree,
                   const int sources[],
                   const int sourceweights[],
                   int outdegree,
                   const int destinations[],
                   const int destweights[],
                   MPIL_Info* info,
                   MPIL_Topo** mpil_topo_ptr)
{
    MPIL_Topo* mpil_topo = (MPIL_Topo*)malloc(sizeof(MPIL_Topo));

    // Copy indegree and outdegree into MPIL_Topo struct
    mpil_topo->indegree  = indegree;
    mpil_topo->outdegree = outdegree;

    // Create copy of sources/destinations in MPIL_Topo struct
    mpil_topo->sources      = NULL;
    mpil_topo->destinations = NULL;

    if (indegree)
    {
        mpil_topo->sources = (int*)malloc(indegree * sizeof(int));
        memcpy(mpil_topo->sources, sources, indegree * sizeof(int));
        if (sourceweights != MPI_UNWEIGHTED)
        {
            mpil_topo->sourceweights = (int*)malloc(indegree * sizeof(int));
            memcpy(mpil_topo->sourceweights, sourceweights, indegree * sizeof(int));
        }
        else
        {
            mpil_topo->sourceweights = MPI_UNWEIGHTED;
        }
    }

    if (outdegree)
    {
        mpil_topo->destinations = (int*)malloc(outdegree * sizeof(int));
        memcpy(mpil_topo->destinations, destinations, outdegree * sizeof(int));

        if (destweights != MPI_UNWEIGHTED)
        {
            mpil_topo->destweights = (int*)malloc(outdegree * sizeof(int));
            memcpy(mpil_topo->destweights, destweights, outdegree * sizeof(int));
        }
        else
        {
            mpil_topo->destweights = MPI_UNWEIGHTED;
        }
    }

    *mpil_topo_ptr = mpil_topo;

    return MPI_SUCCESS;
}
