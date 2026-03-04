#include <stdlib.h>  // For NULL

#include "locality_aware.h"
#include "neighborhood/neighborhood_init.h"
#include "persistent/MPIL_Request.h"

int neighbor_start(MPIL_Request* request)
{
    if (request == NULL)
    {
        return 0;
    }

    int ierr = 0;
    int idx;

    char* send_buffer = NULL;
    int recv_size     = 0;
    if (request->recv_size)
    {
        send_buffer = (char*)(request->sendbuf);
        recv_size   = request->recv_size;
    }

    // Local L sends sendbuf
    if (request->local_L_request != NULL)
    {
        for (int i = 0; i < request->locality->local_L_comm->send_data->size_msgs; i++)
        {
            idx = request->locality->local_L_comm->send_data->indices[i];
            for (int j = 0; j < recv_size; j++)
            {
                request->locality->local_L_comm->send_data->buffer[i * recv_size + j] =
                    send_buffer[idx * recv_size + j];
            }
        }

        MPIL_Start(request->local_L_request);
    }

    // Local S sends sendbuf
    if (request->local_S_request != NULL)
    {
        for (int i = 0; i < request->locality->local_S_comm->send_data->size_msgs; i++)
        {
            idx = request->locality->local_S_comm->send_data->indices[i];

            for (int j = 0; j < recv_size; j++)
            {
                request->locality->local_S_comm->send_data->buffer[i * recv_size + j] =
                    send_buffer[idx * recv_size + j];
            }
        }

        MPIL_Start(request->local_S_request);

        MPI_Status status;
        MPIL_Wait(request->local_S_request, MPI_STATUS_IGNORE);

        // Copy into global->send_data->buffer
        for (int i = 0; i < request->locality->global_comm->send_data->size_msgs; i++)
        {
            idx = request->locality->global_comm->send_data->indices[i];
            for (int j = 0; j < recv_size; j++)
            {
                request->locality->global_comm->send_data->buffer[i * recv_size + j] =
                    request->locality->local_S_comm->recv_data
                        ->buffer[idx * recv_size + j];
            }
        }

    }

    // Global sends buffer in locality, sendbuf in standard
    if (request->n_msgs)
    {
        ierr += MPI_Startall(request->n_msgs, request->requests);
    }

    return ierr;
}
