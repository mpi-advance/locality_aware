#include "../../../../include/communicator/mpil_comm.h"

int MPIL_Comm_win_free(MPIL_Comm* xcomm)
{
    int rank, num_procs;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_size(xcomm->global_comm, &num_procs);

    if (xcomm->win != MPI_WIN_NULL)
    {
        MPI_Win_free(&(xcomm->win));
    }
    if (xcomm->win_array != NULL)
    {
        MPI_Free_mem(xcomm->win_array);
    }
    xcomm->win_bytes      = 0;
    xcomm->win_type_bytes = 0;

    return MPI_SUCCESS;
}
