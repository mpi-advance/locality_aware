#include "communicator/MPIL_Comm.h"

int MPIL_Comm_leader_free(MPIL_Comm* xcomm)
{
    if (xcomm->leader_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&(xcomm->leader_comm));
    }
    if (xcomm->leader_group_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&(xcomm->leader_group_comm));
    }
    if (xcomm->leader_local_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&(xcomm->leader_local_comm));
    }

    return MPI_SUCCESS;
}