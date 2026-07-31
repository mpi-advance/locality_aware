#include <stdlib.h>  // For NULL
#include <string.h>

#include "locality_aware.h"
#include "neighborhood/neighborhood_init.h"
#include "persistent/MPIL_Request.h"

#if defined(GPU)
#include "heterogeneous/gpu_utils.h"
#endif

// Wait for locality-aware requests
// 1. Wait for global
// 2. Start and wait for local_R
// 3. Wait for local_L
// TODO : Currently ignores the status!
int neighbor_wait(MPIL_Request* request, MPI_Status* status)
{
    if (request == NULL)
    {
        return 0;
    }

    int ierr = 0;
    int idx;

    // Global waits for recvs
    if (request->n_msgs)
    {
        ierr += MPI_Waitall(
            request->n_msgs, request->requests, MPI_STATUSES_IGNORE);

        if (request->size_recvs && request->recv_indices)
        {
            for (int i = 0; i < request->size_recvs; i++)
            {
                idx = request->recv_indices[i];

                MPI_Sendrecv((char*)(request->tmp_recvbuf) + (i*request->recv_size),
                        request->recv_size, MPI_BYTE, 0, 0, 
                        (char*)(request->recvbuf) + (idx*request->recv_size),
                        request->recv_size, MPI_BYTE, 0, 0,
                        MPI_COMM_SELF, MPI_STATUS_IGNORE);
            }
        }
    }

    // Wait for local_R recvs
    if (request->local_R_request)
    {
        MPIL_Start(request->local_R_request);
        MPIL_Wait(request->local_R_request, MPI_STATUS_IGNORE);
    }

    // Wait for local_L recvs
    if (request->local_L_request)
    {
        MPIL_Wait(request->local_L_request, MPI_STATUS_IGNORE);
     }

#if defined(GPU)
    if (request->gpu_recvbuf)
    {
#if defined(APU)
        memcpy(request->gpu_recvbuf, request->recvbuf, request->size_recvs);
#else
        int gpu_error;
        gpu_error = gpuMemcpyAsync(request->gpu_recvbuf, request->recvbuf, request->size_recvs, 
                gpuMemcpyHostToDevice, 0);
        gpu_check(gpu_error);
        gpu_error = gpuStreamSynchronize(0);
        gpu_check(gpu_error);
#endif
    }
#endif

    return ierr;
}
