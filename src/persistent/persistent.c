#include "persistent.h"

void init_request(MPIL_Request** request_ptr)
{
    MPIL_Request* request = (MPIL_Request*)malloc(sizeof(MPIL_Request));

    request->locality = NULL;

    request->local_L_n_msgs = 0;
    request->local_S_n_msgs = 0;
    request->local_R_n_msgs = 0;
    request->global_n_msgs  = 0;

    request->local_L_requests = NULL;
    request->local_S_requests = NULL;
    request->local_R_requests = NULL;
    request->global_requests  = NULL;

    request->recv_size  = 0;
    request->block_size = 1;

#ifdef GPU
    request->cpu_sendbuf = NULL;
    request->cpu_recvbuf = NULL;
#endif

    *request_ptr = request;
}

void allocate_requests(int n_requests, MPI_Request** request_ptr)
{
    if (n_requests)
    {
        MPI_Request* request = (MPI_Request*)malloc(sizeof(MPI_Request) * n_requests);
        *request_ptr         = request;
    }
    else
    {
        *request_ptr = NULL;
    }
}

// Starting locality-aware requests
// 1. Start Local_L
// 2. Start and wait for local_S
// 3. Start global
int MPIL_Start(MPIL_Request* request)
{
    if (request == NULL)
    {
        return 0;
    }

    mpix_start_ftn start_function = (mpix_start_ftn)(request->start_function);
    return start_function(request);
}

// Wait for locality-aware requests
// 1. Wait for global
// 2. Start and wait for local_R
// 3. Wait for local_L
// TODO : Currently ignores the status!
int MPIL_Wait(MPIL_Request* request, MPI_Status* status)
{
    if (request == NULL)
    {
        return 0;
    }

    mpix_wait_ftn wait_function = (mpix_wait_ftn)(request->wait_function);
    return wait_function(request, status);
}

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
