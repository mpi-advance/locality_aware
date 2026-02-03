#include <stdlib.h>

#include "communicator/MPIL_Comm.h"
#include "locality_aware.h"

int MPIL_Comm_init(MPIL_Comm** xcomm_ptr, MPI_Comm global_comm)
{
    MPIL_Comm* xcomm   = (MPIL_Comm*)malloc(sizeof(MPIL_Comm));
    xcomm->global_comm = global_comm;

    xcomm->local_comm = MPI_COMM_NULL;
    xcomm->group_comm = MPI_COMM_NULL;

    xcomm->leader_comm       = MPI_COMM_NULL;
    xcomm->leader_group_comm = MPI_COMM_NULL;
    xcomm->leader_local_comm = MPI_COMM_NULL;

    xcomm->neighbor_comm = MPI_COMM_NULL;

    xcomm->win       = MPI_WIN_NULL;
    xcomm->win_array = NULL;
    xcomm->win_bytes = 0;

    xcomm->requests   = NULL;
    xcomm->statuses   = NULL;
    xcomm->n_requests = 0;

    int flag;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &(xcomm->max_tag), &flag);
    xcomm->tag = 126 % xcomm->max_tag;

    xcomm->global_rank_to_local = NULL;
    xcomm->global_rank_to_node  = NULL;
    xcomm->ordered_global_ranks = NULL;

#ifdef GPU
    xcomm->gpus_per_node = 0;
#endif

    *xcomm_ptr = xcomm;

    return MPI_SUCCESS;
}
