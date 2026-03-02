#include <stdlib.h>  // For NULL

#include "locality_aware.h"
#include "persistent/MPIL_Request.h"

#ifdef __cplusplus
extern "C" {
#endif

int MPIL_Wait(MPIL_Request* request, MPI_Status* status)
{
    if (request == NULL)
    {
        return 0;
    }

    return (request->wait_function)(request, status);
}

#ifdef __cplusplus
}
#endif


