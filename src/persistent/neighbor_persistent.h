#ifndef MPI_ADVANCE_NEIGHBOR_INIT_H
#define MPI_ADVANCE_NEIGHBOR_INIT_H

#include "communicator/locality_comm.h"
#include "persistent.h"
#include "neighborhood/neighbor.h"

#ifdef __cplusplus
extern "C"
{
#endif

// Starting locality-aware requests
// 1. Start Local_L
// 2. Start and wait for local_S
// 3. Start global
int neighbor_start(MPIX_Request* request);


// Wait for locality-aware requests
// 1. Wait for global
// 2. Start and wait for local_R
// 3. Wait for local_L
int neighbor_wait(MPIX_Request* request, MPI_Status* status);

#ifdef __cplusplus
}
#endif

#endif
