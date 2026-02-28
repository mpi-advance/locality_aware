#include "communicator/MPIL_Comm.hpp"
#include "communicator/global_comms.hpp"
#include "locality_aware.h"

MPIL_Comm* MPIL_COMM_WORLD;

int MPIL_Init(MPI_Comm world)
{
    if (MPI_COMM_NULL == world)
    {
        world = MPI_COMM_WORLD;
    }

    /* Duplicate World Communicator */
    MPI_Comm_dup(world, &Communicator::WORLD_COMM);

    /* Create MPIL_COMM_WORLD */
    initialize_comm_object(&MPIL_COMM_WORLD, Communicator::WORLD_COMM);
    initialize_topo_communicator(MPIL_COMM_WORLD);
    initialize_rank_mapping(MPIL_COMM_WORLD);

    return MPI_SUCCESS;
}