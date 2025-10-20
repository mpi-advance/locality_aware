#include <stdlib.h>
#include <mpi.h>
#include "neighborhood/MPIL_Topo.h"

int MPIL_Topo_free(MPIL_Topo** mpil_topo_ptr)
{
    MPIL_Topo* mpil_topo = *mpil_topo_ptr;

    if (mpil_topo->indegree)
    {
        free(mpil_topo->sources);
        if (mpil_topo->sourceweights != MPI_UNWEIGHTED)
        {
            free(mpil_topo->sourceweights);
        }
    }

    if (mpil_topo->outdegree)
    {
        free(mpil_topo->destinations);
        if (mpil_topo->destweights != MPI_UNWEIGHTED)
        {
            free(mpil_topo->destweights);
        }
    }
    free(mpil_topo);
    return MPI_SUCCESS;
}
