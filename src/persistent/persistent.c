#include "persistent.h"


int MPIX_Start(MPIX_Request* request)
{
    if (request == NULL)
        return 0;

    mpix_start_ftn start_function = (mpix_start_ftn)(request->start_function);
    return start_function(request);
}

int MPIX_Wait(MPIX_Request* request, MPI_Status* status)
{
    if (request == NULL)
        return 0;

    mpix_wait_ftn wait_function = (mpix_wait_ftn)(request->wait_function);
    return wait_function(request, status);
}

int MPIX_Request_free(MPIX_Request* request)
{
    if (request->local_L_n_msgs)
    {
        for (int i = 0; i < request->local_L_n_msgs; i++)
            MPI_Request_free(&(request->local_L_requests[i]));
        free(request->local_L_requests);
    }
    if (request->local_S_n_msgs)
    {
        for (int i = 0; i < request->local_S_n_msgs; i++)
            MPI_Request_free(&(request->local_S_requests[i]));
        free(request->local_S_requests);
    }
    if (request->local_R_n_msgs)
    {
        for (int i = 0; i < request->local_R_n_msgs; i++)
            MPI_Request_free(&(request->local_R_requests[i]));
        free(request->local_R_requests);
    }
    if (request->global_n_msgs)
    {
        for (int i = 0; i < request->global_n_msgs; i++)
            MPI_Request_free(&(request->global_requests[i]));
        free(request->global_requests);
    }

    // If Locality-Aware
    if (request->locality)
        destroy_locality_comm(request->locality);

    free(request);

    return 0;
}



// Starting locality-aware requests
// 1. Start Local_L
// 2. Start and wait for local_S
// 3. Start global
int neighbor_start(MPIX_Request* request)
{
    if (request == NULL)
        return 0;

    int ierr, idx;

    char* send_buffer = NULL;
    int recv_size = 0;
    if (request->recv_size)
    {
        send_buffer = (char*)(request->sendbuf);
        recv_size = request->recv_size;
    }

    // Local L sends sendbuf
    if (request->local_L_n_msgs)
    {
        for (int i = 0; i < request->locality->local_L_comm->send_data->size_msgs; i++)
        {
            idx = request->locality->local_L_comm->send_data->indices[i];
            for (int j = 0; j < recv_size; j++)
                request->locality->local_L_comm->send_data->buffer[i*recv_size+j] = send_buffer[idx*recv_size+j];
        }
        ierr = MPI_Startall(request->local_L_n_msgs, request->local_L_requests);
    }


    // Local S sends sendbuf
    if (request->local_S_n_msgs)
    {
        for (int i = 0; i < request->locality->local_S_comm->send_data->size_msgs; i++)
        {
            idx = request->locality->local_S_comm->send_data->indices[i];

            for (int j = 0; j < recv_size; j++)
                request->locality->local_S_comm->send_data->buffer[i*recv_size+j] = send_buffer[idx*recv_size+j];
        }

        ierr = MPI_Startall(request->local_S_n_msgs, request->local_S_requests);
        ierr = MPI_Waitall(request->local_S_n_msgs, request->local_S_requests, MPI_STATUSES_IGNORE);


        // Copy into global->send_data->buffer
        for (int i = 0; i < request->locality->global_comm->send_data->size_msgs; i++)
        {
            idx = request->locality->global_comm->send_data->indices[i];
            for (int j = 0; j < recv_size; j++)
                request->locality->global_comm->send_data->buffer[i*recv_size+j] = request->locality->local_S_comm->recv_data->buffer[idx*recv_size+j];
        }
    }

    // Global sends buffer in locality, sendbuf in standard
    if (request->global_n_msgs)
        ierr = MPI_Startall(request->global_n_msgs, request->global_requests);

    return ierr;
}


// Wait for locality-aware requests
// 1. Wait for global
// 2. Start and wait for local_R
// 3. Wait for local_L
// TODO : Currently ignores the status!
int neighbor_wait(MPIX_Request* request, MPI_Status* status)
{
    if (request == NULL)
        return 0;

    int ierr, idx;

    char* recv_buffer = NULL;
    int recv_size = 0;
    if (request->recv_size)
    {
        recv_buffer = (char*)(request->recvbuf); 
        recv_size = request->recv_size;
    }

    // Global waits for recvs
    if (request->global_n_msgs)
    {
        ierr = MPI_Waitall(request->global_n_msgs, request->global_requests, MPI_STATUSES_IGNORE);

        if (request->local_R_n_msgs)
        {
            for (int i = 0; i < request->locality->local_R_comm->send_data->size_msgs; i++)
            {
                idx = request->locality->local_R_comm->send_data->indices[i];
                for (int j = 0; j < recv_size; j++)
                    request->locality->local_R_comm->send_data->buffer[i*recv_size+j] = request->locality->global_comm->recv_data->buffer[idx*recv_size+j];
            }
        }
    }

    // Wait for local_R recvs
    if (request->local_R_n_msgs)
    {
        ierr = MPI_Startall(request->local_R_n_msgs, request->local_R_requests);
        ierr = MPI_Waitall(request->local_R_n_msgs, request->local_R_requests, MPI_STATUSES_IGNORE);

        for (int i = 0; i < request->locality->local_R_comm->recv_data->size_msgs; i++)
        {
            idx = request->locality->local_R_comm->recv_data->indices[i];
            for (int j = 0; j < recv_size; j++)
                recv_buffer[idx*recv_size+j] = request->locality->local_R_comm->recv_data->buffer[i*recv_size+j];
        }
    }

    // Wait for local_L recvs
    if (request->local_L_n_msgs)
    {
        ierr = MPI_Waitall(request->local_L_n_msgs, request->local_L_requests, MPI_STATUSES_IGNORE);

        for (int i = 0; i < request->locality->local_L_comm->recv_data->size_msgs; i++)
        {
            idx = request->locality->local_L_comm->recv_data->indices[i];
            for (int j = 0; j < recv_size; j++)
                recv_buffer[idx*recv_size+j] = request->locality->local_L_comm->recv_data->buffer[i*recv_size+j];
        }
    }

    return ierr;
}



