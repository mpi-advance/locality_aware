#include "locality_aware.h"
#include "persistent/MPIL_Request.h"
#include <stdlib.h> // For NULL

// Starting locality-aware requests
// 1. Start Local_L
// 2. Start and wait for local_S
// 3. Start global
/** @brief wrapper interface for the start function of the request object**/
int MPIL_Start(MPIL_Request* request)
{
    if (request == NULL)
    {
        return 0;
    }

    mpil_start_ftn start_function = (mpil_start_ftn)(request->start_function);
    return start_function(request);
}
