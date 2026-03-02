#include "communicator/MPIL_Comm.hpp"
#include "locality_aware.h"

#ifdef __cplusplus
extern "C" {
#endif

int MPIL_Comm_update_locality(MPIL_Comm* xcomm, int ppn)
{
    /* Don't want the users updating the MPIL_COMM_WORLD */
    if (xcomm == MPIL_COMM_WORLD)
    {
        return MPI_ERR_ARG;
    }

    /* Cleanup normal topology objects */
    MPIL_Comm_topo_free(xcomm);
    /* Re-create the communicators */
    initialize_topo_communicator(xcomm, ppn);
    /* And various internal arrays */
    initialize_rank_mapping(xcomm);

    return MPI_SUCCESS;
}

#ifdef __cplusplus
}
#endif