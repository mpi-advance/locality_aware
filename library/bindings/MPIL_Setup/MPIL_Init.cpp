#include "communicator/global_comms.hpp"
#include "locality_aware.h"

int MPIL_Init(MPI_Comm world)
{
    if (MPI_COMM_NULL == world)
    {
        world = MPI_COMM_WORLD;
    }
    Communicator::initialize_communicators(world);
    return MPI_SUCCESS;
}