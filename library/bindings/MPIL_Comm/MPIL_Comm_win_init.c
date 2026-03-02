#include "communicator/MPIL_Comm.h"
#include "locality_aware.h"

int MPIL_Comm_win_init(MPIL_Comm* xcomm, int bytes, int type_bytes)
{
    xcomm->win_bytes      = bytes;
    xcomm->win_type_bytes = type_bytes;
    MPI_Alloc_mem(xcomm->win_bytes, MPI_INFO_NULL, &(xcomm->win_array));
    MPI_Win_create(xcomm->win_array,
                   xcomm->win_bytes,
                   xcomm->win_type_bytes,
                   MPI_INFO_NULL,
                   xcomm->global_comm,
                   &(xcomm->win));

    return MPI_SUCCESS;
}
