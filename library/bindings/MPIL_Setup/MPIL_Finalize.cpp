#include "communicator/global_comms.hpp"
#include "locality_aware.h"

#ifdef __cplusplus
extern "C" {
#endif

int MPIL_Finalize()
{
    MPIL_Comm_free(&MPIL_COMM_WORLD);

    Communicator::cached_local_comms.clear();
    Communicator::cached_group_comms.clear();

    return MPI_SUCCESS;
}

#ifdef __cplusplus
}
#endif