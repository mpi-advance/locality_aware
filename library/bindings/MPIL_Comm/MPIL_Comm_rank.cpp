#include "communicator/MPIL_Comm.hpp"
#include "locality_aware.h"

int MPIL_Comm_rank(MPIL_Comm* xcomm, int* rank)
{
    MPI_Comm_rank(xcomm->global_comm, rank);
    return MPI_SUCCESS;
}

int MPIL_Comm_local_rank(MPIL_Comm* xcomm, int* local_rank)
{
    MPI_Comm_rank(xcomm->local_comm, local_rank);
    return MPI_SUCCESS;
}

int MPIL_Comm_group_rank(MPIL_Comm* xcomm, int* group_rank)
{
    MPI_Comm_rank(xcomm->group_comm, group_rank);
    return MPI_SUCCESS;
}
