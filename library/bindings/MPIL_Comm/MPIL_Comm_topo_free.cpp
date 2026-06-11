#include <stdlib.h>

#include <iostream>

#include "communicator/MPIL_Comm.hpp"
#include "locality_aware.h"

int MPIL_Comm_topo_free(MPIL_Comm* xcomm)
{
    if (xcomm == MPIL_COMM_WORLD)
    {
        return MPI_SUCCESS;
    }

    free(xcomm->global_rank_to_local);
    xcomm->global_rank_to_local = NULL;

    free(xcomm->global_rank_to_node);
    xcomm->global_rank_to_node = NULL;

    free(xcomm->ordered_global_ranks);
    xcomm->ordered_global_ranks = NULL;

    return MPI_SUCCESS;
}
