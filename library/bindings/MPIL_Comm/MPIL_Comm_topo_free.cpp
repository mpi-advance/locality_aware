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
    
    if (xcomm->local_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&(xcomm->local_comm));
        xcomm->local_comm = MPI_COMM_NULL;
    }
    if (xcomm->group_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&(xcomm->group_comm));
        xcomm->group_comm = MPI_COMM_NULL;
    }

    if (xcomm->global_rank_to_local != NULL)
    {
        free(xcomm->global_rank_to_local);
        xcomm->global_rank_to_local = NULL;
    }
    if (xcomm->global_rank_to_node != NULL)
    {
        free(xcomm->global_rank_to_node);
        xcomm->global_rank_to_node = NULL;
    }
    if (xcomm->ordered_global_ranks != NULL)
    {
        free(xcomm->ordered_global_ranks);
        xcomm->ordered_global_ranks = NULL;
    }

    return MPI_SUCCESS;
}
