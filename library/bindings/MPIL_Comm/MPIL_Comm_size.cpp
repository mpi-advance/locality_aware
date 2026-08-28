#include "communicator/MPIL_Comm.hpp"
#include "locality_aware.h"

int MPIL_Comm_size(MPIL_Comm* xcomm, int* size)
{
    MPI_Comm_size(xcomm->global_comm, size);
    return MPI_SUCCESS;
}

int MPIL_Comm_local_size(MPIL_Comm* xcomm, int* local_size)
{
    MPI_Comm_size(xcomm->local_comm, local_size);
    return MPI_SUCCESS;
}

int MPIL_Comm_group_size(MPIL_Comm* xcomm, int* group_size)
{
    MPI_Comm_size(xcomm->group_comm, group_size);
    return MPI_SUCCESS;
}
