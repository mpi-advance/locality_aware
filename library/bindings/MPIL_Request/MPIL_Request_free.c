#include <stdlib.h>

#include "locality_aware.h"
#include "persistent/MPIL_Request.h"
#ifdef GPU
#include "heterogeneous/gpu_utils.h"
#endif

int MPIL_Request_free(MPIL_Request** request_ptr)
{
    MPIL_Request* request = *request_ptr;

    if (request->local_L_request != NULL)
    {
        MPIL_Request_free(&(request->local_L_request));
    }

    if (request->local_S_request != NULL)
    {
        MPIL_Request_free(&(request->local_S_request));
    }

    if (request->local_R_request != NULL)
    {
        MPIL_Request_free(&(request->local_R_request));
    }

    if (request->n_msgs)
    {
        for (int i = 0; i < request->n_msgs; i++)
        {
            MPI_Request_free(&(request->requests[i]));
        }
        free(request->requests);
    }

    // If Locality-Aware
    if (request->locality != NULL)
    {
        destroy_locality_comm(request->locality);
    }

// TODO : for safety, may want to check if allocated with malloc?
#ifdef GPU  // Assuming cpu buffers allocated in pinned memory
    int ierr;
    if (request->cpu_sendbuf)
    {
        ierr = gpuFreeHost(request->cpu_sendbuf);
        gpu_check(ierr);
    }
    if (request->cpu_recvbuf)
    {
        ierr = gpuFreeHost(request->cpu_recvbuf);
        gpu_check(ierr);
    }
#endif

    free(request);

    return 0;
}
