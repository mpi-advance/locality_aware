#include "../../../include/communicator/mpil_comm.h"



int MPIL_Comm_win_init(MPIL_Comm* xcomm, int bytes, int type_bytes)
{
    int rank, num_procs;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_size(xcomm->global_comm, &num_procs);

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
