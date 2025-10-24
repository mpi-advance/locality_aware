#include <stdlib.h>

#include "communicator/MPIL_Comm.h"
#include "locality_aware.h"

int MPIL_Comm_free(MPIL_Comm** xcomm_ptr)
{
    MPIL_Comm* xcomm = *xcomm_ptr;

    if (xcomm->n_requests > 0)
    {
        free(xcomm->requests);
    }

    if (xcomm->neighbor_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&(xcomm->neighbor_comm));
    }

    MPIL_Comm_topo_free(xcomm);
    MPIL_Comm_leader_free(xcomm);
    MPIL_Comm_win_free(xcomm);
    MPIL_Comm_device_free(xcomm);

    free(xcomm);

    return MPI_SUCCESS;
}
