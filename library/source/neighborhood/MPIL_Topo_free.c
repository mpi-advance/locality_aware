#include "locality_aware.h"

#include <string.h>

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
