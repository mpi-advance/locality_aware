#include "persistent.h"

int MPIX_Request_init(MPIX_Request** request_ptr)
{
    MPIX_Request* request = (MPIX_Request*)malloc(sizeof(MPIX_Request));

    request->locality = NULL;

    request->locked_win=0;
    request->local_L_n_msgs = 0;
    request->local_S_n_msgs = 0;
    request->local_R_n_msgs = 0;
    request->global_n_msgs = 0;

    request->local_L_requests = NULL;
    request->local_S_requests = NULL;
    request->local_R_requests = NULL;
    request->global_requests = NULL;

    request->recv_size = 0;

    request->xcomm = NULL;
    request->sdispls = NULL;
    request->put_displs = NULL;
    request->send_sizes = NULL;
    request->recv_sizes = NULL;
    request->n_puts = 0;

    *request_ptr = request;

    return MPI_SUCCESS;
}

int allocate_requests(int n_requests, MPI_Request** request_ptr)
{
    if (n_requests)
    {
        MPI_Request* request = (MPI_Request*)malloc(sizeof(MPI_Request)*n_requests);
        *request_ptr = request;
    }
    else *request_ptr = NULL;

    return MPI_SUCCESS;
}

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
    int rank;
    MPI_Comm_rank(request->xcomm->global_comm,&rank);
    if (request->locked_win)
    {
        MPI_Win_unlock(rank,request->xcomm->win); // MGFD: Release Local Exclusive Lock, this allows other process to safely put data. 

    }
    
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
   // printf("global_num_msg %d\n",request->global_n_msgs);
    //fflush(stdout);
    if (request->global_n_msgs)
    {
        for (int i = 0; i < request->global_n_msgs; i++)
       {     
        printf("print before fail; i=%d,ptr=%p\n",i,request->global_requests[i]);
        fflush(stdout);
        
        MPI_Request_free(&(request->global_requests[i]));
        
        }
       
        free(request->global_requests);
        request->global_requests=0;
    }

    // If Locality-Aware
    if (request->locality)
        destroy_locality_comm(request->locality);

    if (request->sdispls)
        free(request->sdispls);
    if (request->put_displs)
        free(request->put_displs);
    if (request->send_sizes)
        free(request->send_sizes);
    if (request->recv_sizes)
        free(request->recv_sizes);

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


// Batched Persistent Alltoall Operation
int batch_start(MPIX_Request* request)
{
    if (request == NULL)
        return 0;

    MPI_Startall(2*request->batch, request->global_requests);

    return MPI_SUCCESS;
}

int batch_wait(MPIX_Request* request, MPI_Status* status)
{
    int n = request->batch;
    MPI_Waitall(2*n, request->global_requests, MPI_STATUSES_IGNORE);

    int num_procs = request->global_n_msgs / 2;

    for (int i = n; i < num_procs; i += n)
    {
        if (i + n > num_procs)
            n = num_procs - i;
            

        MPI_Startall(2*n, &(request->global_requests[2*i]));
        MPI_Waitall(2*n, &(request->global_requests[2*i]), MPI_STATUSES_IGNORE);
    }
        

    return MPI_SUCCESS;
}



int rma_start(MPIX_Request* request)
{
    int rank;

    MPI_Comm_rank(request->xcomm->global_comm, &rank);  // Get the rank of the process
    
    char* send_buffer = (char*)(request->sendbuf);
    char* recv_buffer = (char*)(request->recvbuf);
        // printf("Process %d entering rma_start\n", rank);

    //MPI_Barrier(request->xcomm->global_comm);  // synchronized before proceeding.

    MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOPRECEDE, request->xcomm->win);
    for (int i = 0; i < request->n_puts; i++)
    {
         MPI_Put(&(send_buffer[request->sdispls[i]]), request->send_sizes[i], MPI_CHAR,
                 i, request->put_displs[i], request->recv_sizes[i], MPI_CHAR, request->xcomm->win);
    }   
        //printf("Process %d leaving rma_start\n", rank);  

    return MPI_SUCCESS;
}


int rma_wait(MPIX_Request* request, MPI_Status* status)
{
    
   
    MPI_Win_fence(MPI_MODE_NOPUT|MPI_MODE_NOSUCCEED, request->xcomm->win);

    
    return MPI_SUCCESS;
}


int rma_lock_start(MPIX_Request* request)
{
    int rank;
    //printf("*************hey just entered rma_lock_start");
    //fflush(stdout);
    MPI_Comm_rank(request->xcomm->global_comm, &rank);  // Get the rank of the process

    char* send_buffer = (char*)(request->sendbuf);
    char* recv_buffer = (char*)(request->recvbuf); 
    //printf("*************329");
    //fflush(stdout);
    MPI_Win_unlock(rank, request->xcomm->win); // MGFD: Release Local Exclusive Lock, this allows other process to safely put data. 

    //printf("******************333");
   // fflush(stdout);
       // Lock the window for all processes
    MPI_Win_lock_all(0, request->xcomm->win);
   
        
    int request_count = 0;
   // printf("*************after MPI_Win_lock_all***");
   // fflush(stdout);

    //  non-blocking MPI_Rput 
    
    for (int i = 0; i < request->n_puts; i++) {
        if (request->send_sizes[i] > 0) {
            MPI_Rput(&request->sendbuf[request->sdispls[i]],
                request->send_sizes[i], 
                MPI_CHAR,
                i,
                request->put_displs[i], 
                request->send_sizes[i],
                MPI_CHAR,
                request->xcomm->win,
                &(request->global_requests[request_count++]));
        }
    }
    request->global_n_msgs=request_count;


MPI_Waitall(request->global_n_msgs, request->global_requests, MPI_STATUSES_IGNORE);
    
/*
    for (int i = 0; i < request->n_puts; i++) {
        if (request->send_sizes[i] > 0) {
            MPI_Put(&request->sendbuf[request->sdispls[i]],
                    request->send_sizes[i],
                    MPI_CHAR,
                    i,
                    request->put_displs[i],
                    request->send_sizes[i],
                    MPI_CHAR,
                    request->xcomm->win);
        }
    }
    // tracking 0 requests now
    request->global_n_msgs = 0;
*/

   // printf("lock_start nputs=%d, request count= %d\n",request->n_puts, request_count);
    //fflush(stdout);

    //printf("*************after Rputs***");
    //fflush(stdout);
    // Waiting for all non-blocking operations to complete 
    /*newly added */
   

   // printf("*************after MPI_Waitall***");
   // fflush(stdout);    
    
    return MPI_SUCCESS;
}

    
     
int rma_lock_wait(MPIX_Request* request, MPI_Status* status)
{
   
    int rank;  // MGFD: We should really save this off at init, but we need this here for the exclusive lock.
    MPI_Comm_rank(request->xcomm->global_comm, &rank);
    
   
    MPI_Win_unlock_all(request->xcomm->win);
    
    //MPI_Win_flush_all(request->xcomm->win); //MGFD: We're already doing the unlock and lock below so the flush is not needed
    //printf(" MPI_Win_unlock_all******************13");
    //fflush(stdout); 
    for (int i = 0; i < request->global_n_msgs; i++) {

        MPI_Request_free(&request->global_requests[i]);
      
      }
      
      
    
   // MPI_Barrier(request->xcomm->global_comm); // MGFD: Barrier would be needed if we were relying of flush, but we're not, so we don't need it.
    //printf("******************14");
    //fflush(stdout);

    /*newly commented out */
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, request->xcomm->win); // MGFD: this makes the buffer go into an consistent state, and therefore is safe to access by the user.
    
    //printf("******************15");
    //fflush(stdout);
    //memcpy(recv_buffer, request->xcomm->win_array, request->recv_size);

    return MPI_SUCCESS;
}


  


