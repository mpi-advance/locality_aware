#include "communicator/global_comms.hpp"
#include "locality_aware.h"

int MPIL_Finalize()
{
    MPIL_Comm_free(&MPIL_COMM_WORLD);
    return MPI_SUCCESS;
}