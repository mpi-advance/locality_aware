#include "locality_aware.h"
#include "persistent/MPIL_Request.h"

#ifdef __cplusplus
extern "C" {
#endif

int MPIL_Request_reorder(MPIL_Request* request, int value)
{
    request->reorder = value;
    return MPI_SUCCESS;
}

#ifdef __cplusplus
}
#endif