#include <stdlib.h>  // For NULL

#include "locality_aware.h"
#include "persistent/MPIL_Request.h"

int MPIL_Start(MPIL_Request* request)
{
    if (request == NULL)
    {
        return 0;
    }

    return (request->start_function)(request);
}
