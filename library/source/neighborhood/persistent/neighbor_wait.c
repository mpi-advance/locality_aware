#include "neighborhood/neighborhood_init.h"
#include "persistent/MPIL_Request.h"
#include <stdlib.h> // For NULL

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

    char* recv_buffer = NULL;
    int recv_size     = 0;
    if (request->recv_size)
    {
        recv_buffer = (char*)(request->recvbuf);
        recv_size   = request->recv_size;
    }

    // Global waits for recvs
    if (request->global_n_msgs)
    {
        ierr += MPI_Waitall(
            request->global_n_msgs, request->global_requests, MPI_STATUSES_IGNORE);

        if (request->local_R_n_msgs)
        {
            for (int i = 0; i < request->locality->local_R_comm->send_data->size_msgs;
                 i++)
            {
                idx = request->locality->local_R_comm->send_data->indices[i];
                for (int j = 0; j < recv_size; j++)
                {
                    request->locality->local_R_comm->send_data
                        ->buffer[i * recv_size + j] =
                        request->locality->global_comm->recv_data
                            ->buffer[idx * recv_size + j];
                }
            }
        }
    }

    // Wait for local_R recvs
    if (request->local_R_n_msgs)
    {
        ierr += MPI_Startall(request->local_R_n_msgs, request->local_R_requests);
        ierr += MPI_Waitall(
            request->local_R_n_msgs, request->local_R_requests, MPI_STATUSES_IGNORE);

        for (int i = 0; i < request->locality->local_R_comm->recv_data->size_msgs; i++)
        {
            idx = request->locality->local_R_comm->recv_data->indices[i];
            for (int j = 0; j < recv_size; j++)
            {
                recv_buffer[idx * recv_size + j] =
                    request->locality->local_R_comm->recv_data->buffer[i * recv_size + j];
            }
        }
    }

    // Wait for local_L recvs
    if (request->local_L_n_msgs)
    {
        ierr += MPI_Waitall(
            request->local_L_n_msgs, request->local_L_requests, MPI_STATUSES_IGNORE);

        for (int i = 0; i < request->locality->local_L_comm->recv_data->size_msgs; i++)
        {
            idx = request->locality->local_L_comm->recv_data->indices[i];
            for (int j = 0; j < recv_size; j++)
            {
                recv_buffer[idx * recv_size + j] =
                    request->locality->local_L_comm->recv_data->buffer[i * recv_size + j];
            }
        }
    }

    return ierr;
}
