#include <stdlib.h>

#include "communicator/MPIL_Comm.hpp"
#include "locality_aware.h"

#ifdef __cplusplus
extern "C" {
#endif

int MPIL_Comm_topo_init(MPIL_Comm* xcomm)
{
    if (xcomm == MPIL_COMM_WORLD)
    {
        return MPI_SUCCESS;
    }

    initialize_topo_communicator(xcomm);
    initialize_rank_mapping(xcomm);

    return MPI_SUCCESS;
}

#ifdef __cplusplus
}
#endif
