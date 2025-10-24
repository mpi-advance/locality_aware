#include <cstdlib>

#include "communicator/MPIL_Info.h"
#include "locality_aware.h"

int MPIL_Info_free(MPIL_Info** info_ptr)
{
    MPIL_Info* xinfo = *info_ptr;
    free(xinfo);

    return MPI_SUCCESS;
}