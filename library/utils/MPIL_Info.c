#include "utils.h"
#include "mpi.h"

/ MPIL Info Object Routines
int MPIL_Info_init(MPIL_Info** info_ptr)
{
    MPIL_Info* xinfo            = (MPIL_Info*)malloc(sizeof(MPIL_Info));
    xinfo->crs_num_initialized  = 0;
    xinfo->crs_size_initialized = 0;

    *info_ptr = xinfo;

    return MPI_SUCCESS;
}

int MPIL_Info_free(MPIL_Info** info_ptr)
{
    MPIL_Info* xinfo = *info_ptr;
    free(xinfo);

    return MPI_SUCCESS;
}
