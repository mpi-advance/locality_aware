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

    if (request->tmpbuf != NULL)
    {
        request->free_ftn(request->tmpbuf);
    }

    // Added with MPI_Comm_dup, so need freed
    if (request->global_comm != MPI_COMM_NULL)
        MPI_Comm_free(&(request->global_comm));
    if (request->local_comm != MPI_COMM_NULL)
        MPI_Comm_free(&(request->local_comm));

#ifdef GPU  
    //For now, we have always allocated these with MPIL_Alloc 
    
    // Can't free sendbuf (const), so free tmp_sendbuf
    if (request->gpu_sendbuf)
        MPIL_Free(request->tmp_gpubuf);

    if (request->gpu_recvbuf)
        MPIL_Free(request->recvbuf);
#endif

    free(request);

    return 0;
}

#ifdef __cplusplus
}
#endif
