#include "../../../../include/communicator/mpil_comm.h"


int MPIL_Comm_topo_free(MPIL_Comm* xcomm)
{
    if (xcomm->local_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&(xcomm->local_comm));
    }
    if (xcomm->group_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&(xcomm->group_comm));
    }

    if (xcomm->global_rank_to_local != NULL)
    {
        free(xcomm->global_rank_to_local);
    }
    if (xcomm->global_rank_to_node != NULL)
    {
        free(xcomm->global_rank_to_node);
    }
    if (xcomm->ordered_global_ranks != NULL)
    {
        free(xcomm->ordered_global_ranks);
    }

    return MPI_SUCCESS;
}
