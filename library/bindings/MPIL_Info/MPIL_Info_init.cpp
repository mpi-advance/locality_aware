#include "locality_aware.h"
#include "communicator/MPIL_Info.h"
#include <cstdlib>

// MPIL Info Object Routines
int MPIL_Info_init(MPIL_Info** info_ptr)
{
    MPIL_Info* xinfo            = (MPIL_Info*)malloc(sizeof(MPIL_Info));
    xinfo->crs_num_initialized  = 0;
    xinfo->crs_size_initialized = 0;

    *info_ptr = xinfo;

    return MPI_SUCCESS;
}
