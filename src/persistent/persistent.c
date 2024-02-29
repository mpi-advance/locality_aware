#include "persistent.h"


// Starting locality-aware requests
// 1. Start Local_L
// 2. Start and wait for local_S
// 3. Start global
int MPIX_Start(MPIX_Request* request)
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
    {
        if (request->reorder)
        {
            ierr = MPI_Startall(request->global_n_sends, &(request->global_requests[request->global_n_recvs]));
        }
        else
            ierr = MPI_Startall(request->global_n_msgs, request->global_requests);
    }

    return ierr;
}


// Wait for locality-aware requests
// 1. Wait for global
// 2. Start and wait for local_R
// 3. Wait for local_L
// TODO : Currently ignores the status!
int MPIX_Wait(MPIX_Request* request, MPI_Status* status)
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
        if (request->reorder)
            ierr = reorder_wait(request);
        else
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



int reorder_wait(MPIX_Request* request)
{
    if (request == NULL)
        return 0;
    
    int num_procs, rank;
    MPI_Comm_size(request->xcomm->neighbor_comm, &num_procs);
    MPI_Comm_rank(request->xcomm->neighbor_comm, &rank);

    request->reorder = 0;

    int ierr, idx, proc, count;
    int recv_size;
    MPI_Status status;

    MPI_Type_size(request->recvtype, &recv_size);
        //printf("Rank %d should recv %d msgs\n", rank, request->global_n_recvs);
        
    char* recv_buffer = NULL;
    if (request->global_n_recvs)
    {
        recv_buffer = (char*)(request->recvbuf); 

        int* orig_ptr = (int*)malloc(num_procs*sizeof(int));
        for (int i = 0; i < request->global_n_recvs; i++)
        {
            proc = request->recv_procs[i];
            orig_ptr[proc] =  request->rdispls[i];
        }

        for (int i = 0; i < request->global_n_recvs; i++)
        {
            MPI_Request_free(&(request->global_requests[i]));
            MPI_Probe(MPI_ANY_SOURCE, request->tag, request->xcomm->neighbor_comm, &status);
            proc = status.MPI_SOURCE;
            MPI_Get_count(&status, request->recvtype, &count);
            MPI_Recv_init(&(recv_buffer[orig_ptr[proc]*recv_size]), count, request->recvtype, 
                    proc, request->tag, request->xcomm->neighbor_comm, 
                    &(request->global_requests[i]));
            MPI_Start(&(request->global_requests[i]));
            MPI_Wait(&(request->global_requests[i]), &status);
            //printf("Rank %d recvd i %d from proc %d, count %d, at recv_buffer[%d]\n", rank, i, proc, count, orig_ptr[proc]*recv_size); 
        }

        free(orig_ptr);
    }

    if (request->global_n_sends)
        ierr = MPI_Waitall(request->global_n_sends, 
                &(request->global_requests[request->global_n_recvs]), 
                MPI_STATUSES_IGNORE);

    MPI_Barrier(request->xcomm->global_comm);
    printf("Rank %d past barrier!\n", rank);
    return ierr;
}


int MPIX_Request_init(MPIX_Request** request_ptr, MPIX_Comm* xcomm)
{
    MPIX_Request* request = (MPIX_Request*)malloc(sizeof(MPIX_Request));

    request->locality = NULL;

    request->local_L_n_msgs = 0;
    request->local_S_n_msgs = 0;
    request->local_R_n_msgs = 0;
    request->global_n_msgs = 0;

    request->local_L_requests = NULL;
    request->local_S_requests = NULL;
    request->local_R_requests = NULL;
    request->global_requests = NULL;

    request->sendbuf = NULL;
    request->recvbuf = NULL;
    request->recv_size = 0;
    request->global_n_sends = 0;
    request->global_n_recvs = 0;
    request->tag = 0;
    request->recv_procs = NULL;
    request->rdispls = NULL;

    request->reorder = 0;

    request->xcomm = xcomm;

    *request_ptr = request;
}

int MPIX_Request_free(MPIX_Request** request_ptr)
{
    MPIX_Request* request = *request_ptr;

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

int store_sources(MPIX_Request* request, int n_procs, int* procs)
{
    if (n_procs)
    {
        if (request->recv_procs != NULL)
            free(request->recv_procs);
        request->recv_procs = (int*)malloc(n_procs*sizeof(int));
        for (int i = 0; i < n_procs; i++)
            request->recv_procs[i] = procs[i];
    }
}

int store_rdispls(MPIX_Request* request, int n_rdispls, int* rdispls)
{
    if (n_rdispls)
    {
        if (request->rdispls != NULL)
            free(request->rdispls);
        request->rdispls = (int*)malloc(n_rdispls*sizeof(int));
        for (int i = 0; i < n_rdispls; i++)
            request->rdispls[i] = rdispls[i];
    }
}
