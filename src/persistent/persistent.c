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
    //new
    request->is_local = NULL; 
    request->lockedall_wins = NULL;
    request->remote_targets = NULL;  // list of off-node ranks we send to
    request->n_remote = 0;
    request->local_targets = NULL;   // list of on-node ranks we send to
    request->n_local = 0;

     /*
     * NEW for //ONE (ready to receive) signaling
     *  */
    request->sig_win       = MPI_WIN_NULL;
    request->signals       = NULL;
    request->rtr_flags     = NULL;
    request->done_flags    = NULL;
    request->send_peers    = NULL;
    request->recv_peers    = NULL;
    request->num_send_peers = 0;
    request->num_recv_peers = 0;
    request->epoch=0;
    
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
    
     if (request->lockedall_wins)
    {
    

        MPI_Win_unlock_all(request->sig_win);
        MPI_Win_unlock_all(request->xcomm->win);	

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
        //printf("print before fail; i=%d,ptr=%p\n",i,request->global_requests[i]);
        //fflush(stdout);
        
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

     /* NEW frees */
    if (request->is_local)
        free(request->is_local);
    if (request->remote_targets)
        free(request->remote_targets);
    if (request->local_targets)
        free(request->local_targets);

    

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


int rma_start_han(MPIX_Request* request)
{
    int rank;
    MPI_Comm_rank(request->xcomm->global_comm, &rank);

    const char* send_buffer = (const char*)(request->sendbuf);
   

    MPI_Barrier(request->xcomm->global_comm);

    MPI_Win_fence(MPI_MODE_NOPRECEDE, request->xcomm->win);

    /* --------- remote ranks first (off-node) --------- */
    for (int idx = 0; idx < request->n_remote; ++idx) {
        int i      = request->remote_targets[idx];
        int nbytes = request->send_sizes[i];
        if (nbytes == 0) continue;

        MPI_Put((const void*)((const char*)request->sendbuf + request->sdispls[i]),
                nbytes, MPI_BYTE,
                i, (MPI_Aint)request->put_displs[i],
                nbytes, MPI_BYTE,
                request->xcomm->win);
    }

    /* --------- local ranks second (same node) -------- */
    for (int idx = 0; idx < request->n_local; ++idx) {
        int i      = request->local_targets[idx];
        int nbytes = request->send_sizes[i];
        if (nbytes == 0) continue;

        MPI_Put((const void*)((const char*)request->sendbuf + request->sdispls[i]),
                nbytes, MPI_BYTE,
                i, (MPI_Aint)request->put_displs[i],
                nbytes, MPI_BYTE,
                request->xcomm->win);
    }

    return MPI_SUCCESS;
}



int rma_start(MPIX_Request* request)
{
    int rank;

    MPI_Comm_rank(request->xcomm->global_comm, &rank);  // Get the rank of the process
    
    const char*  send_buffer = (const char* )(request->sendbuf);
    const char*  recv_buffer = (const char* )(request->recvbuf);
    //     printf("Process %d entering rma_start\n", rank);

    MPI_Barrier(request->xcomm->global_comm);  

    

    MPI_Win_fence(MPI_MODE_NOPRECEDE, request->xcomm->win);
    for (int i = 0; i < request->n_puts; ++i) {
        int nbytes = request->send_sizes[i];      // bytes iam sending to i
        if (nbytes == 0) continue;

        MPI_Put((const void*)((const char*)request->sendbuf + request->sdispls[i]),
                nbytes, MPI_BYTE,
                i, (MPI_Aint)request->put_displs[i],
                nbytes, MPI_BYTE,
                request->xcomm->win);
    }

    //MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOPRECEDE, request->xcomm->win);
   /* MPI_Win_fence(MPI_MODE_NOPRECEDE, request->xcomm->win);
    for (int i = 0; i < request->n_puts; i++)
    {
        
         // Skip if no data to send to this target
    if (request->send_sizes[i] == 0 || request->recv_sizes[i] == 0)
    continue;

//printf("Rank %d: sending %d bytes to Rank %d; Rank %d expects %d bytes from Rank %d\n",
  //     rank, request->send_sizes[i], i,
    //   i, request->recv_sizes[i], i);


         // printf("Process %d starts puts in winfence's rma_start\n", rank);
         MPI_Put(&(send_buffer[request->sdispls[i]]), request->send_sizes[i], MPI_BYTE,
                 i, request->put_displs[i], request->recv_sizes[i], MPI_BYTE, request->xcomm->win);


                  
    }   */


  
  //      printf("Process %d leaving rma_start\n", rank);  

    return MPI_SUCCESS;
}


int rma_wait(MPIX_Request* request, MPI_Status* status)
{
    int rank;

    MPI_Comm_rank(request->xcomm->global_comm, &rank);  // Get the rank of the process


   //MPI_Barrier(request->xcomm->global_comm);
//printf("[rma_wait] Rank %d entering fence\n", rank); fflush(stdout);

    
    MPI_Win_fence(MPI_MODE_NOSUCCEED, request->xcomm->win);
    
//printf("[rma_wait] Rank %d leaving fence\n", rank); fflush(stdout);
   

    return MPI_SUCCESS;
}

/*
rma_lock_start

*/






int rma_lock_start(MPIX_Request* request)
{
    int rank;
    
    MPI_Comm_rank(request->xcomm->global_comm, &rank);  // Get the rank of the process
   // printf("Process %d entering rma_lock_start\n", rank);
    //fflush(stdout);

    const char*  send_buffer = (const char* )(request->sendbuf);
    const char*  recv_buffer = (const char* )(request->recvbuf); 
    //printf("*************329");
    //fflush(stdout);
    MPI_Win_unlock(rank, request->xcomm->win); // MGFD: Release Local Exclusive Lock, this allows other process to safely put data. 

    //printf("******************333");
   // fflush(stdout);
       
    //for (int i = 0; i < request->n_puts; ++i)
       // request->global_requests[i] = MPI_REQUEST_NULL;
    
       
    MPI_Win_lock_all(0, request->xcomm->win);
   
        
    //int request_count = 0;
   // printf("*************after MPI_Win_lock_all***");
   // fflush(stdout);

    //  non-blocking MPI_Rput 
    /*
    for (int i = 0; i < request->n_puts; i++) {
        if (request->send_sizes[i] > 0) {
            MPI_Rput(&request->sendbuf[request->sdispls[i]],
                request->send_sizes[i], 
                 MPI_BYTE,
                i,
                request->put_displs[i], 
                request->send_sizes[i],
                 MPI_BYTE,
                request->xcomm->win,
                &(request->global_requests[request_count++]));
        }
    }
    request->global_n_msgs=request_count;


MPI_Waitall(request->global_n_msgs, request->global_requests, MPI_STATUSES_IGNORE);
    // start MPI_Put

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
 
*/


//MPI_Aint is large enough to store any valid memory address, so it won’t overflow on 64-bit systems.



    for (int i = 0; i < request->n_puts; ++i) {
    int nbytes = request->send_sizes[i];      // bytes iam sending to i
    if (nbytes == 0) continue;

    MPI_Put((const void*)((const char*)request->sendbuf + request->sdispls[i]),
            nbytes, MPI_BYTE,
            i, (MPI_Aint)request->put_displs[i],
            nbytes, MPI_BYTE,
            request->xcomm->win);
    }
     //tracking 0 requests now
    //request->global_n_msgs = 0; 


   // printf("lock_start nputs=%d, request count= %d\n",request->n_puts, request_count);
    //fflush(stdout);

    //printf("*************after Rputs***");
    //fflush(stdout);
    // Waiting for all non-blocking operations to complete 
    
   

    //printf("Process %d leaving rma_lock_start\n", rank);
    //fflush(stdout);    
    
    return MPI_SUCCESS;
}


int rma_lock_start_han(MPIX_Request* request)
{
    int rank;
    
    MPI_Comm_rank(request->xcomm->global_comm, &rank);  // Get the rank of the process
   // printf("Process %d entering rma_lock_start\n", rank);
  // fflush(stdout);

    const char*  send_buffer = (const char* )(request->sendbuf);
    const char*  recv_buffer = (const char* )(request->recvbuf); 
    //printf("*************329");
    //fflush(stdout);
    //
  //   MPI_Barrier(request->xcomm->global_comm);
    MPI_Win_unlock(rank, request->xcomm->win); // MGFD: Release Local Exclusive Lock, this allows other process to safely put data. 


    // MPI_Barrier(request->xcomm->global_comm);
    //printf("******************333");
   // fflush(stdout);
       
    //for (int i = 0; i < request->n_puts; ++i)
       // request->global_requests[i] = MPI_REQUEST_NULL;
    
       
    MPI_Win_lock_all(0, request->xcomm->win);
      

//MPI_Aint is large enough to store any valid memory address, so it won’t overflow on 64-bit systems.



     /* --------- remote ranks first (off-node) --------- */
    for (int idx = 0; idx < request->n_remote; ++idx) {
        int i      = request->remote_targets[idx];
        int nbytes = request->send_sizes[i];
        if (nbytes == 0) continue;

        MPI_Put((const void*)((const char*)request->sendbuf + request->sdispls[i]),
                nbytes, MPI_BYTE,
                i, (MPI_Aint)request->put_displs[i],
                nbytes, MPI_BYTE,
                request->xcomm->win);
    }

    /* --------- local ranks second (same node) -------- */
    for (int idx = 0; idx < request->n_local; ++idx) {
        int i      = request->local_targets[idx];
        int nbytes = request->send_sizes[i];
        if (nbytes == 0) continue;

        MPI_Put((const void*)((const char*)request->sendbuf + request->sdispls[i]),
                nbytes, MPI_BYTE,
                i, (MPI_Aint)request->put_displs[i],
                nbytes, MPI_BYTE,
                request->xcomm->win);
    }
     //tracking 0 requests now
    //request->global_n_msgs = 0; 


   // printf("lock_start nputs=%d, request count= %d\n",request->n_puts, request_count);
    //fflush(stdout);

    //printf("*************after Rputs***");
    //fflush(stdout);
    // Waiting for all non-blocking operations to complete 
    
   

    //printf("Process %d leaving rma_lock_start\n", rank);
    //fflush(stdout);    
    
    return MPI_SUCCESS;
}

/*
rma_lock_start_han ends
*/

int rma_lock_wait_han(MPIX_Request* request, MPI_Status* status)
{
	
    int rank; MPI_Comm_rank(request->xcomm->global_comm, &rank);
 //printf("Process %d starting rma_start\n", rank);

    MPI_Win_unlock_all(request->xcomm->win);

    /* Ensure every process has finished their access epochs  */
   MPI_Barrier(request->xcomm->global_comm);//removed the segfault

    

    // Now re-acquire the exclusive self lock so user code can safely read recvbuf 
    // MGFD: this makes the buffer go into a consistent state, and therefore is safe to access by the user.
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, request->xcomm->win);
    
     //printf("process %d leaving", rank);
    return MPI_SUCCESS;
}

int rma_lock_wait(MPIX_Request* request, MPI_Status* status)
{

    int rank; MPI_Comm_rank(request->xcomm->global_comm, &rank);
// printf("Process %d starting rma_start\n", rank);

    MPI_Win_unlock_all(request->xcomm->win);

    /* Ensure every process has finished their access epochs  */
    MPI_Barrier(request->xcomm->global_comm);//removed the segfault



    // Now re-acquire the exclusive self lock so user code can safely read recvbuf
    // MGFD: this makes the buffer go into an consistent state, and therefore is safe to access by the user.
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, request->xcomm->win);

  //   printf("process %d leaving", rank);
    return MPI_SUCCESS;
}
//===============================================
     
int rma_start_RTS(MPIX_Request* request)
{
    int rank, num_procs;
    MPI_Comm_rank(request->xcomm->global_comm, &rank);
    MPI_Comm_size(request->xcomm->global_comm, &num_procs);

    const char* send_buffer = (const char*)(request->sendbuf);

    
    /* MPI_Win_lock_all moved to INIT */

    /* Used epoch counter so we don't need to clear flags every time */
      int epoch	= request->epoch += 1;

    const int one = 1;

    /* ---------------------------------------------------------------------------------
     * posting, incrementing rtr using MPI_Accumulate
	 * If A will be sending data to B
	 * process B calls MPI_Accumulate to increment(+1) A's memory of rtr flags @ displ B(rtr_flags[B])
    *thats to say:
     * If I (rank B) expect to receive from src(A), I notify src by
     * incrementing src's(A) rtr_flags[rankB].
     *
     * Then, src(A) polls its local rtr_flags[rank(B)]->in the second loop (I removed MPI_Get because it was over the network).
     *-------------------------------------------  */
	 //Go over everybody who will be sending me rtr flags(they send(notify) me siginals now, will send them data in next loop)
    
    
    printf("[rank %d]before accumulate:\n", rank);
    	fflush(stdout);

    for (int k = 0; k < request->num_recv_peers; k++) {
        int src = request->recv_peers[k];

        /* target = src(A), update slot OR position "rank( B)" in src's(A) rtr_flags[] */
        MPI_Accumulate(&one, 1, MPI_INT,
                       src,
                       (MPI_Aint)rank,   /* (disp_unit = sizeof(int) not bytes like in data win) */
                       1, MPI_INT,
                       MPI_SUM,
                       request->sig_win);
    }

    /* I didnt put any flushes in here since I  moved them to WAIT.
	 *It should be noted that at this point the code hangs, there are no print statements seen at this point,
	 *printed after accumulate, but when I tried to flush it worked but again got stack(hang at the next point).
	*/
   // MPI_Win_flush_all(request->sig_win);

     printf("[rank %d]After accumulate:\n",rank);
     	fflush(stdout);

    /* ---------------------------------------------------------------------------
     * now trying the dynamic start of transfers
     *
     * where src(A) is checking whether it has rtr_flags of the destinations it will send too :)(Poll is local: request->rtr_flags[dst]
     * ---------------------------------------- */


	    //
 int remaining = request->num_send_peers;//Has remaining processes that havent seen done flags

    while (remaining > 0) {
	    MPI_Win_sync(request->sig_win);
  //data phase
 printf("[rank %d]before data phase:\n", rank);
 	fflush(stdout);
        for (int i = 0; i < request->num_send_peers; ++i) 
		{
            int dst = request->send_peers[i];

            // ready when dst has incremented my rtr_flags[dst]up to the current epoch 
            if (request->rtr_flags[dst] < epoch)
		    continue;
			

                int nbytes = request->send_sizes[dst];
                if (nbytes > 0) 
				{
                    MPI_Put(send_buffer + request->sdispls[dst],
                            nbytes, MPI_BYTE,
                            dst,
                            (MPI_Aint)request->put_displs[dst],
                            nbytes, MPI_BYTE,
                            request->xcomm->win);//Putting
                }
		

	   request->rtr_flags[dst] = epoch - 1;

            remaining--;
            if (remaining == 0) break;


             } 
    }	
		
  printf("[rank %d]After Data Phase:\n", rank);
	fflush(stdout); 
        
    

    return MPI_SUCCESS;
}



int rma_wait_RTS(MPIX_Request* request, MPI_Status* status)
{
    int rank, num_procs;
    MPI_Comm_rank(request->xcomm->global_comm, &rank);
    MPI_Comm_size(request->xcomm->global_comm, &num_procs);

    int epoch = request->epoch;
    const int one = 1;

    printf("[rank %d]Starting the Wait phase:\n", rank);
        fflush(stdout);

    MPI_Win_flush_all(request->xcomm->win);

     printf("After completing data phase:\n");
        fflush(stdout);

    /* ---------------------------------------------------------
     *  signaling DONE to each destination I put data. 
     *      
     *       done_flags[src] is stored at displacement (num_procs + src) becoz any doneflags will be stored after all the rtr flags which end @numprocs-1,  
     *
     * At dst(B), we want to increment dst.done_flags[rank(A)].
     * so A calls MPI_Accumulate(+1) on B to let it know that iam done sending
     * --------------------------------------------------------- */

	 printf("before accumulating in Wait :\n");
        fflush(stdout);
    //Going over everybody I sent, put data too, increment there doneflag of me[rank] by one, so that they know it that Iam done.
    for (int i = 0; i < request->num_send_peers; i++) {
        int dst = request->send_peers[i];

        MPI_Accumulate(&one, 1, MPI_INT,
                       dst,(num_procs + rank),
                       1, MPI_INT, MPI_SUM,
                       request->sig_win);
    }

    // Complete done siginals
    MPI_Win_flush_all(request->sig_win);

     printf("After accumulating(doneflags) loop in wait:\n");
        fflush(stdout);
    //
 int remaining = request->num_recv_peers;//Has remaining processes that havent seen done flags

 printf("before polling in wait:\n");
        fflush(stdout);  
 while (remaining > 0) {
        MPI_Win_sync(request->sig_win);
	
	for (int k = 0; k < request->num_recv_peers; k++) {
	
	int src = request->recv_peers[k]; 

	if (request->done_flags[src] >= epoch) {
                request->done_flags[src] = epoch - 1; //
    	        remaining--;
                if (remaining == 0) break;
            }
        }
    }		
 printf("After polling in wait:\n");
        fflush(stdout);

    

    return MPI_SUCCESS;
}





