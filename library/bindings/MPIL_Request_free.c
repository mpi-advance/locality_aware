//#include "locality_aware.h"
#include "persistent/MPIL_Request.h"





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
