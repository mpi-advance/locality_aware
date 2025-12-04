#include <stdlib.h>

#include "locality_aware.h"
#include "persistent/MPIL_Request.h"
#include "utils/MPIL_Alloc.h"
#ifdef GPU
#include "heterogeneous/gpu_utils.h"
#endif

int MPIL_Request_free(MPIL_Request** request_ptr)
{
    MPIL_Request* request = *request_ptr;

    if (request->local_L_n_msgs)
    {
        for (int i = 0; i < request->local_L_n_msgs; i++)
        {
            MPI_Request_free(&(request->local_L_requests[i]));
        }
        free(request->local_L_requests);
    }
    if (request->local_S_n_msgs)
    {
        for (int i = 0; i < request->local_S_n_msgs; i++)
        {
            MPI_Request_free(&(request->local_S_requests[i]));
        }
        free(request->local_S_requests);
    }
    if (request->local_R_n_msgs)
    {
        for (int i = 0; i < request->local_R_n_msgs; i++)
        {
            MPI_Request_free(&(request->local_R_requests[i]));
        }
        free(request->local_R_requests);
    }
    if (request->global_n_msgs)
    {
        for (int i = 0; i < request->global_n_msgs; i++)
        {
            MPI_Request_free(&(request->global_requests[i]));
        }
        free(request->global_requests);
    }

    // If Locality-Aware
    if (request->locality != NULL)
    {
        destroy_locality_comm(request->locality);
    }

    if (request->tmpbuf != NULL)
    {
        request->free_ftn(request->tmpbuf);
    }

    if (request->global_comm != MPI_COMM_NULL)
        MPI_Comm_free(&(request->global_comm));
    if (request->local_comm != MPI_COMM_NULL)
        MPI_Comm_free(&(request->local_comm));

// TODO : for safety, may want to check if allocated with malloc?
#ifdef GPU  // Assuming cpu buffers allocated in pinned memory
    int ierr;
    if (request->cpu_sendbuf)
    {
        ierr = gpuHostFree(request->cpu_sendbuf);
        gpu_check(ierr);
    }
    if (request->cpu_recvbuf)
    {
        ierr = gpuHostFree(request->cpu_recvbuf);
        gpu_check(ierr);
    }
#endif

    free(request);

    return 0;
}
