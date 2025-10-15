#include "locality_aware.h"
#include "communicator/MPIL_Info.h"
#include <cstdlib>

int MPIL_Info_free(MPIL_Info** info_ptr)
{
    MPIL_Info* xinfo = *info_ptr;
    free(xinfo);

    return MPI_SUCCESS;
}