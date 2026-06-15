#include <stdlib.h>  // For NULL
#include <string.h>

#include "locality_aware.h"
#include "neighborhood/neighborhood_init.h"
#include "persistent/MPIL_Request.h"

int neighbor_start(MPIL_Request* request)
{
    if (request == NULL)
    {
        return 0;
    }

#if defined(GPU)
    int gpu_error;
    if (request->gpu_sendbuf)
    {
#if defined(APU)
        memcpy(request->tmp_gpubuf, request->gpu_sendbuf, request->size_sends);
#else
        gpu_error = gpuMemcpyAsync(request->tmp_gpubuf, request->gpu_sendbuf, request->size_sends, 
                gpuMemcpyDeviceToHost, 0);
        gpu_check(gpu_error);
        gpu_error = gpuStreamSynchronize(0);
        gpu_check(gpu_error);
#endif
    }
#endif


    int ierr = 0;
    int idx;

    // Local L sends sendbuf
    if (request->local_L_request != NULL)
    {
        MPIL_Start(request->local_L_request);
    }

    // Local S sends sendbuf
    if (request->local_S_request != NULL)
    {
        MPIL_Start(request->local_S_request);
        MPIL_Wait(request->local_S_request, MPI_STATUS_IGNORE);
    }

    // Global sends buffer in locality, sendbuf in standard
    if (request->n_msgs)
    {
        if (request->size_sends && request->send_indices)
        {
            for (int i = 0; i < request->size_sends; i++)
            {
                idx = request->send_indices[i];
                memcpy((char*)(request->tmp_sendbuf) + (i*request->send_size), 
                        (char*)(request->sendbuf) + (idx*request->send_size), 
                        request->send_size);        
            }
        }
        ierr += MPI_Startall(request->n_msgs, request->requests);
    }

    return ierr;
}
