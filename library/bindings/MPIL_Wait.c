#include "../../include/persistent/persistent.h"

// Wait for locality-aware requests
// 1. Wait for global
// 2. Start and wait for local_R
// 3. Wait for local_L
// TODO : Currently ignores the status!
/** @brief wrapper interface for the wait function of the request object**/
int MPIL_Wait(MPIL_Request* request, MPI_Status* status)
{
    if (request == NULL)
    {
        return 0;
    }

    mpix_wait_ftn wait_function = (mpix_wait_ftn)(request->wait_function);
    return wait_function(request, status);
}