#include "communicator/global_comms.hpp"
#include "locality_aware.h"

int MPIL_Finalize()
{
    Communicator::teardown_communicators();
    return MPI_SUCCESS;
}