#include <stdlib.h>

#include "locality_aware.h"
#include "persistent/MPIL_Request.h"
#ifdef GPU
#include "heterogeneous/gpu_utils.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

int MPIL_Request_free(MPIL_Request** request_ptr)
{
    MPIL_Request* request = *request_ptr;

    /** Free any local request objects.  In these objects,
     * sendbuf and recvbuf were malloc'd, so they must be freed */
    if (request->local_L_request != NULL)
    {
        MPIL_Request_free(&(request->local_L_request));
        request->local_L_request = NULL;
    }
    if (request->local_S_request != NULL)
    {
        MPIL_Request_free(&(request->local_S_request));
        request->local_S_request = NULL;
    }
    if (request->local_R_request != NULL)
    {
        MPIL_Request_free(&(request->local_R_request));
        request->local_R_request = NULL;
    }

    if (request->n_msgs)
    {
        for (int i = 0; i < request->n_msgs; i++)
        {
            MPI_Request_free(&(request->requests[i]));
        }
        free(request->requests);
        request->n_msgs = 0;
    }

    if (request->size_sends)
    {
        free(request->tmp_sendbuf);
        request->tmp_sendbuf = NULL;

        free(request->send_indices);
        request->send_indices = NULL;

        request->size_sends = 0;
    }
    if (request->size_recvs)
    {
        free(request->tmp_recvbuf);
        request->tmp_recvbuf = NULL;

        free(request->recv_indices);
        request->recv_indices = NULL;

        request->size_recvs = 0;
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

#ifdef __cplusplus
}
#endif
