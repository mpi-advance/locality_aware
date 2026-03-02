#include <stdlib.h>  // For NULL

#include "locality_aware.h"
#include "persistent/MPIL_Request.h"

#ifdef __cplusplus
extern "C" {
#endif

int MPIL_Start(MPIL_Request* request)
{
    if (request == NULL)
    {
        return 0;
    }

    return (request->start_function)(request);
}

#ifdef __cplusplus
}
#endif
