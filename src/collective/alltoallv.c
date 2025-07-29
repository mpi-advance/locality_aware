#include "alltoallv.h"
#include <string.h>
#include <math.h>
#include "utils.h"
//#include "/g/g92/enamug/install/include/caliper/cali.h"
#include "/g/g92/enamug/my_caliper_install/include/caliper/cali.h"
/**************************************************
 * Locality-Aware Point-to-Point Alltoallv
 * Same as PMPI_Alltoall (no load balancing)
 *  - Aggregates messages locally to reduce 
 *      non-local communciation
 *  - First redistributes on-node so that each
 *      process holds all data for a subset
 *      of other nodes
 *  - Then, performs inter-node communication
 *      during which each process exchanges
 *      data with their assigned subset of nodes
 *  - Finally, redistribute received data
 *      on-node so that each process holds
 *      the correct final data
 *  - To be used when sizes are relatively balanced
 *  - For load balacing, use persistent version
 *      - Load balacing is too expensive for 
 *          non-persistent Alltoallv
 *************************************************/
int MPI_Alltoallv(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPI_Comm comm)
{
    return alltoallv_pairwise(
        sendbuf,
        sendcounts,
        sdispls,
        sendtype,
        recvbuf,
        recvcounts,
        rdispls,
        recvtype,
        comm);
}

int MPIX_Alltoallv(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* mpi_comm)
{
    return alltoallv_waitany(sendbuf,
        sendcounts,
        sdispls,
        sendtype,
        recvbuf,
        recvcounts,
        rdispls,
        recvtype,
        mpi_comm->global_comm);
}


int alltoallv_pairwise(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPI_Comm comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    int tag = 103044;
    int send_proc, recv_proc;
    int send_pos, recv_pos;
    MPI_Status status;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    memcpy(
        recvbuf + (rdispls[rank] * recv_size),
        sendbuf + (sdispls[rank] * send_size), 
        sendcounts[rank] * send_size);        

    // Send to rank + i
    // Recv from rank - i
    for (int i = 1; i < num_procs; i++)
    {
        send_proc = rank + i;
        if (send_proc >= num_procs)
            send_proc -= num_procs;
        recv_proc = rank - i;
        if (recv_proc < 0)
            recv_proc += num_procs;

        send_pos = sdispls[send_proc] * send_size;
        recv_pos = rdispls[recv_proc] * recv_size;

        MPI_Sendrecv(sendbuf + send_pos, sendcounts[send_proc], sendtype, send_proc, tag,
                recvbuf + recv_pos, recvcounts[recv_proc], recvtype, recv_proc, tag,
                comm, &status);
    }

    return 0;
}

int alltoallv_nonblocking(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPI_Comm comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    int tag = 103044;
    int send_proc, recv_proc;
    int send_pos, recv_pos;
    MPI_Status status;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    MPI_Request* requests = (MPI_Request*)malloc(2*(num_procs-1)*sizeof(MPI_Request));

    memcpy(
        recvbuf + (rdispls[rank] * recv_size),
        sendbuf + (sdispls[rank] * send_size), 
        sendcounts[rank] * send_size);        

    // For each step i
    // exchange among procs stride (i+1) apart
    for (int i = 1; i < num_procs; i++)
    {
        send_proc = rank + i;
        if (send_proc >= num_procs)
            send_proc -= num_procs;
        recv_proc = rank - i;
        if (recv_proc < 0)
            recv_proc += num_procs;

        send_pos = sdispls[send_proc] * send_size;
        recv_pos = rdispls[recv_proc] * recv_size;

        MPI_Isend(sendbuf + send_pos, sendcounts[send_proc], sendtype, send_proc, tag,
                comm, &(requests[i-1]));
        MPI_Irecv(recvbuf + recv_pos, recvcounts[recv_proc], recvtype, recv_proc, tag,
                comm, &(requests[num_procs+i-2]));
    }

    MPI_Waitall(2*(num_procs-1), requests, MPI_STATUSES_IGNORE);

    free(requests);

    return 0;
}



int alltoallv_rma_winfence(const void* sendbuf,
                  const int sendcounts[],
                  const int sdispls[],
                  MPI_Datatype sendtype,
                  void* recvbuf,
                  const int recvcounts[],
                  const int rdispls[],
                  MPI_Datatype recvtype,
                  MPIX_Comm* xcomm)
{
    int rank, num_procs;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_size(xcomm->global_comm, &num_procs);

    char* send_buffer = (char*)(sendbuf);
    char* recv_buffer = (char*)(recvbuf);

    int send_type_size, recv_type_size;
    MPI_Type_size(sendtype, &send_type_size);
    MPI_Type_size(recvtype, &recv_type_size);


    int* rdispls_dist =(int*)malloc(num_procs*sizeof(int));
    MPI_Alltoall(rdispls, 1, MPI_INT, rdispls_dist, 1, MPI_INT, xcomm->global_comm);
    // Calculate the total bytes needed for receiving
    int total_recv_bytes = 0;
    for (int i = 0; i < num_procs; i++) {
        total_recv_bytes += recvcounts[i] * recv_type_size;
    }

    
    if (xcomm->win_bytes != total_recv_bytes || xcomm->win_type_bytes != 1) {
        MPIX_Comm_win_free(xcomm);
    }

    // Initialize window only if it hasn't been initialized
    if (xcomm->win == MPI_WIN_NULL) {
        MPI_Win_create(recv_buffer, total_recv_bytes, 1, MPI_INFO_NULL, xcomm->global_comm, &xcomm->win);
    }

   
CALI_MARK_BEGIN("start_rma_Win_fence_region");

    MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOPRECEDE, xcomm->win);//epoch starts

CALI_MARK_END("start_rma_Win_fence_region");


CALI_MARK_BEGIN("put_region_rma_Winfence");
    
    for (int i = 0; i < num_procs; i++) {
        if (sendcounts[i] > 0) {
            MPI_Put(&(send_buffer[sdispls[i] * send_type_size]),
                    sendcounts[i] * send_type_size,
                    MPI_CHAR,
                    i,
                    rdispls_dist[i] * recv_type_size,
                    sendcounts[i] * send_type_size,
                    MPI_CHAR,
                    xcomm->win);
        }
    }

CALI_MARK_END("put_region_rma_Winfence");

   

CALI_MARK_BEGIN("Final_rma_Win_fence_region");

    // Final synchronization
    MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOSUCCEED, xcomm->win);

CALI_MARK_END("Final_rma_Win_fence_region");

CALI_MARK_BEGIN("Final_barrier_Win_fence_region");

    MPI_Barrier(xcomm->global_comm);
CALI_MARK_END("Final_barrier_Win_fence_region");

free(rdispls_dist);
    return MPI_SUCCESS;
}


/**************** */

int alltoallv_rma_winfence_init(const void* sendbuf,
    const int sendcounts[],
    const int sdispls[],
    MPI_Datatype sendtype,
    void* recvbuf,
    const int recvcounts[],
    const int rdispls[],
    MPI_Datatype recvtype,
    MPIX_Comm* xcomm,
    MPIX_Info* xinfo,
    MPIX_Request** request_ptr)
{
    int rank, num_procs;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_size(xcomm->global_comm, &num_procs);

    MPIX_Request* request;
    MPIX_Request_init(&request);
        
    request->start_function = rma_start;
    request->wait_function = rma_wait;

    request->sendbuf = sendbuf;
    request->recvbuf = recvbuf;

    int send_type_size, recv_type_size;
    MPI_Type_size(sendtype, &send_type_size);
    MPI_Type_size(recvtype, &recv_type_size);

   
    int total_recv_bytes = 0;
    for (int i = 0; i < num_procs; i++) {
        total_recv_bytes += recvcounts[i] * recv_type_size;
    }

    
    if (xcomm->win_bytes != total_recv_bytes || xcomm->win_type_bytes != 1) {
        MPIX_Comm_win_free(xcomm);
    }

    // Initialize window only if it hasn't been initialized
    if (xcomm->win == MPI_WIN_NULL) {
        MPI_Win_create(recvbuf, total_recv_bytes, 1, MPI_INFO_NULL, xcomm->global_comm, &xcomm->win);
    }

   
   request->n_puts = num_procs;

   request->xcomm = xcomm;
   request->sdispls = (int*)malloc(num_procs*sizeof(int));
   request->send_sizes = (int*)malloc(num_procs*sizeof(int));
   request->recv_sizes = (int*)malloc(num_procs*sizeof(int));
   request->recv_size = total_recv_bytes;

      
   request->put_displs =(int*)malloc(num_procs*sizeof(int));

   MPI_Alltoall(rdispls, 1, MPI_INT, request->put_displs, 1, MPI_INT, xcomm->global_comm);

   for (int i = 0; i < num_procs; i++)
   {
       request->sdispls[i] = sdispls[i]*send_type_size;
       request->send_sizes[i] = sendcounts[i] * send_type_size;
       request->recv_sizes[i] = recvcounts[i] * recv_type_size;
       request->put_displs[i] *= recv_type_size; 
   }

*request_ptr = request;

    return MPI_SUCCESS;
}

int alltoallv_rma_lock_init(const void* sendbuf,
    const int sendcounts[],
    const int sdispls[],
    MPI_Datatype sendtype,
    void* recvbuf,
    const int recvcounts[],
    const int rdispls[],
    MPI_Datatype recvtype,
    MPIX_Comm* xcomm,
    MPIX_Info* xinfo,
    MPIX_Request** request_ptr)
{
    int rank, num_procs;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_size(xcomm->global_comm, &num_procs);

    printf("%d: *****Iam in rma lock init*************1\n",rank); fflush(stdout);
    MPIX_Request* request;
    MPIX_Request_init(&request);
    
   // printf("******************1");
    request->start_function = rma_lock_start;
   // printf("******************2");
    request->wait_function = rma_lock_wait;
   // printf("******************3");
    //request->global_n_msgs = 0;
    //allocate_requests(num_procs, &(request->global_requests));
    request->sendbuf = sendbuf;
    request->recvbuf = recvbuf;

    int send_type_size, recv_type_size;
    MPI_Type_size(sendtype, &send_type_size);
    MPI_Type_size(recvtype, &recv_type_size);

   
    int total_recv_bytes = 0;
    for (int i = 0; i < num_procs; i++) {
        total_recv_bytes += recvcounts[i] * recv_type_size;
    }

    printf("%d: *****Iam at line %d rma lock init*************1\n",rank,__LINE__); fflush(stdout);
    if (xcomm->win_bytes != total_recv_bytes || xcomm->win_type_bytes != 1) {
        printf("%d: *****Iam at line %d rma lock init************* Before win_free\n",rank,__LINE__); fflush(stdout);
        MPIX_Comm_win_free(xcomm);
        printf("%d: *****Iam at line %d rma lock init************* After win_free\n",rank,__LINE__); fflush(stdout);
    
    }
    printf("%d: *****Iam at line %d rma lock init*************1\n",rank,__LINE__); fflush(stdout);

    // MGFD: This needs refactoring. The window should belong to the request object, not the communicator. As this is written, a second persistant alltoallv on a communicator will fail, because the data will be improperly placed. 
    // Initialize window only if it hasn't been initialized
    if (xcomm->win == MPI_WIN_NULL) {
        printf("%d: *****Iam at line %d rma lock init************* Before win_create\n",rank,__LINE__); fflush(stdout);
        MPI_Win_create(recvbuf, total_recv_bytes, 1, MPI_INFO_NULL, xcomm->global_comm, &xcomm->win);
        printf("%d: *****Iam at line %d rma lock init************* After win_create\n",rank,__LINE__); fflush(stdout);
    }
    //printf("******************4");
    printf("%d: *****Iam at line %d rma lock init*************1\n",rank,__LINE__); fflush(stdout);
    request->xcomm = xcomm;
    
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, request->xcomm->win); //MGFD: To ensure that the recv buffer is not modified before it is valid to do so, we need to do an exclusive self-lock here. 

    //printf("******************5");
    printf("%d: *****Iam at line %d rma lock init*************1\n",rank,__LINE__); fflush(stdout);
   request->n_puts = num_procs;

   request->sdispls = (int*)malloc(num_procs*sizeof(int));
   request->send_sizes = (int*)malloc(num_procs*sizeof(int));
   request->recv_sizes = (int*)malloc(num_procs*sizeof(int));
   request->recv_size = total_recv_bytes;

   printf("%d: *****Iam at line %d rma lock init*************1\n",rank,__LINE__); fflush(stdout);
   request->put_displs =(int*)malloc(num_procs*sizeof(int));
   MPI_Alltoall(rdispls, 1, MPI_INT, request->put_displs, 1, MPI_INT, xcomm->global_comm);

   //printf("******************6");
   printf("%d: *****Iam at line %d rma lock init*************1\n",rank,__LINE__); fflush(stdout);

   for (int i = 0; i < num_procs; i++)
   {
       request->sdispls[i] = sdispls[i]*send_type_size;
       request->send_sizes[i] = sendcounts[i] * send_type_size;
       request->recv_sizes[i] = recvcounts[i] * recv_type_size;
       request->put_displs[i] *= recv_type_size; 
   }
  
  
   //printf("******************7");
*request_ptr = request;
    //printf("******************8");
    printf("%d: *****Iam exiting rma lock init*************\n",rank); fflush(stdout);
    return MPI_SUCCESS;
}



//*************************************************************************************************** 

/*Win Flush similar implemetation to alltoallv_rma_winlock but  I used MPI_Put(blocking)  */
/*
int alltoallv_rma_winflush(const void* sendbuf,
    const int sendcounts[],
    const int sdispls[],
    MPI_Datatype sendtype,
    void* recvbuf,
    const int recvcounts[],
    const int rdispls[],
    MPI_Datatype recvtype,
    MPIX_Comm* xcomm)
{
int rank, num_procs;
MPI_Comm_rank(xcomm->global_comm, &rank);
MPI_Comm_size(xcomm->global_comm, &num_procs);

char* send_buffer = (char*)(sendbuf);
char* recv_buffer = (char*)(recvbuf);

int send_type_size, recv_type_size;
MPI_Type_size(sendtype, &send_type_size);
MPI_Type_size(recvtype, &recv_type_size);

// Calculate the total bytes needed for receiving
int total_recv_bytes = 0;
for (int i = 0; i < num_procs; i++) {
total_recv_bytes += recvcounts[i] * recv_type_size;
}


if (xcomm->win_bytes != total_recv_bytes || xcomm->win_type_bytes != 1) {
MPIX_Comm_win_free(xcomm);
}

if (xcomm->win == MPI_WIN_NULL) {
MPI_Win_create(recv_buffer, total_recv_bytes, 1, MPI_INFO_NULL, xcomm->global_comm, &xcomm->win);
}

// Locking the window for all processes using SHARED mode
CALI_MARK_BEGIN("start_Win_lock__used_in_rma_winflush");
MPI_Win_lock_all(0, xcomm->win);
CALI_MARK_END("start_Win_lock__used_in_rma_winflush");

// Perform RMA put operations
CALI_MARK_BEGIN("put_region_rma_Winflush");
for (int i = 0; i < num_procs; i++) {
if (sendcounts[i] > 0) {
MPI_Put(&(send_buffer[sdispls[i] * send_type_size]),
sendcounts[i] * send_type_size,
MPI_CHAR,
i,
rdispls[rank] * recv_type_size,
recvcounts[rank] * recv_type_size,
MPI_CHAR,
xcomm->win);
}
}
CALI_MARK_END("put_region_rma_Winflush");

// Flush all RMA operations 
CALI_MARK_BEGIN("flush_all_region_rma_winflush");
MPI_Win_flush_all(xcomm->win);
CALI_MARK_END("flush_all_region_rma_winflush");

// Unlock window after all operations are completed
CALI_MARK_BEGIN("Final_Win_unlock_all_region_rma_winflush");
MPI_Win_unlock_all(xcomm->win);
CALI_MARK_END("Final_Win_unlock_all_region_rma_winflush");

MPI_Barrier(xcomm->global_comm);

return MPI_SUCCESS;
}
*/
//uses non-blocking MPI_Rput 


int alltoallv_rma_winlock(const void* sendbuf,
                          const int sendcounts[],
                          const int sdispls[],
                          MPI_Datatype sendtype,
                          void* recvbuf,
                          const int recvcounts[],
                          const int rdispls[],
                          MPI_Datatype recvtype,
                          MPIX_Comm* xcomm)
{
    int rank, num_procs;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_size(xcomm->global_comm, &num_procs);

    char* send_buffer = (char*)(sendbuf);
    char* recv_buffer = (char*)(recvbuf);

    int send_type_size, recv_type_size;
    MPI_Type_size(sendtype, &send_type_size);
    MPI_Type_size(recvtype, &recv_type_size);
    
    int* rdispls_dist =(int*)malloc(num_procs*sizeof(int));
    MPI_Alltoall(rdispls, 1, MPI_INT, rdispls_dist, 1, MPI_INT, xcomm->global_comm);

    int total_recv_bytes = 0;
    for (int i = 0; i < num_procs; i++) {
        total_recv_bytes += recvcounts[i] * recv_type_size;
    }

    
    if (xcomm->win == MPI_WIN_NULL || xcomm->win_bytes != total_recv_bytes || xcomm->win_type_bytes != 1) {
        if (xcomm->win != MPI_WIN_NULL) {
            MPI_Win_free(&xcomm->win);
        }
        MPI_Win_create(recv_buffer, total_recv_bytes, 1, MPI_INFO_NULL, xcomm->global_comm, &xcomm->win);
        xcomm->win_bytes = total_recv_bytes;
        xcomm->win_type_bytes = 1;
    }

    CALI_MARK_BEGIN("start_Win_lock__used_in_rma_winlock");

    // Lock the window for all processes
    MPI_Win_lock_all(0, xcomm->win);
    //local window-> each acceses the target exclusively

    CALI_MARK_END("start_Win_lock__used_in_rma_winlock");

    CALI_MARK_BEGIN("Put_region__used_in_rma_winlock");

    
    MPI_Request* requests = (MPI_Request*)malloc(num_procs * sizeof(MPI_Request));
    int request_count = 0;

    //  non-blocking MPI_Rput 
    for (int i = 0; i < num_procs; i++) {
        if (sendcounts[i] > 0) {
            MPI_Rput(&(send_buffer[sdispls[i] * send_type_size]),
                     sendcounts[i] * send_type_size,
                     MPI_CHAR,
                     i,
                     rdispls_dist[i] * recv_type_size,
                     sendcounts[i] * recv_type_size,
                     MPI_CHAR,
                     xcomm->win,
                     &requests[request_count++]);
        }
    }

    // Waiting for all non-blocking operations to complete 
    //MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);

    
    free(requests);

    CALI_MARK_END("Put_region__used_in_rma_winlock");

    CALI_MARK_BEGIN("Final_Win_flush_region__used_in_rma_winlock");

    // to complete RMA operations 
    MPI_Win_flush_all(xcomm->win);

   
    //CALI_MARK_END("Final_Win_flush_region__used_in_rma_winlock");
   
    CALI_MARK_BEGIN("Final_Win_unlock_region__used_in_rma_winlock");

    //unlocking
    MPI_Win_unlock_all(xcomm->win);

    CALI_MARK_END("Final_Win_unlock_region__used_in_rma_winlock");
    
    MPI_Barrier(xcomm->global_comm);

    free(rdispls_dist);

    return MPI_SUCCESS;
}


//winlock newly optimized version following the intructions below instructions;
/*
can you add lockall flags

For i in all processes

    MpiAccumulate +1 flags rank i

end for

unlock flags 

while myflag != numprocs //spin check
*/
////New New



int alltoallv_rma_newly_winlock(const void* sendbuf,
    const int sendcounts[],
    const int sdispls[],
    MPI_Datatype sendtype,
    void* recvbuf,
    const int recvcounts[],
    const int rdispls[],
    MPI_Datatype recvtype,
    MPIX_Comm* xcomm)
{
int rank, num_procs;
MPI_Comm_rank(xcomm->global_comm, &rank);
MPI_Comm_size(xcomm->global_comm, &num_procs);

char* send_buffer = (char*)(sendbuf);
char* recv_buffer = (char*)(recvbuf);

int send_type_size, recv_type_size;
MPI_Type_size(sendtype, &send_type_size);
MPI_Type_size(recvtype, &recv_type_size);

int total_recv_bytes = 0;
for (int i = 0; i < num_procs; i++) {
total_recv_bytes += recvcounts[i] * recv_type_size;
}

if (xcomm->win_bytes != total_recv_bytes || xcomm->win_type_bytes != 1) {
MPIX_Comm_win_free(xcomm);
}

if (xcomm->win == MPI_WIN_NULL) {
MPI_Win_create(recv_buffer, total_recv_bytes, 1, MPI_INFO_NULL, xcomm->global_comm, &xcomm->win);
}


int my_flag = 0;
int* flag_array;
MPI_Win flag_win;
MPI_Alloc_mem(1 * sizeof(int), MPI_INFO_NULL, &flag_array);
memset(flag_array, 0, 1 * sizeof(int)); 
MPI_Win_create(flag_array, 1 * sizeof(int), sizeof(int), MPI_INFO_NULL, xcomm->global_comm, &flag_win);

int one = 1;


//  MPI_Put
MPI_Win_lock_all(0, xcomm->win);
for (int i = 0; i < num_procs; i++) {
if (sendcounts[i] > 0) {
//printf("%e ", send_buffer);

MPI_Put(&(send_buffer[sdispls[i] * send_type_size]),
      sendcounts[i] * send_type_size,
      MPI_CHAR,
      i,
      rdispls[rank] * recv_type_size,
      recvcounts[rank] * recv_type_size,
      MPI_CHAR,
      xcomm->win);

}
}
MPI_Win_flush_all( xcomm->win);
MPI_Win_unlock_all( xcomm->win);

// Locking for accumulation
MPI_Win_lock_all(0, flag_win);
for (int i = 0; i < num_procs; i++) {
MPI_Accumulate(&one, 1, MPI_INT, i, 0, 1, MPI_INT, MPI_SUM, flag_win);
}
//unlock
MPI_Win_flush_all(flag_win);
MPI_Win_unlock_all(flag_win);

//MPI_Barrier(xcomm->global_comm);
// Checking if the accumulate completed @ all processes
//MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, flag_win);


MPI_Request dummy_req; int dummy_flag = 0;
MPI_Irecv(NULL, 0, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, xcomm->global_comm, &dummy_req);

//lock

MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, flag_win);

 while(flag_array[0] < num_procs){
    MPI_Win_unlock(rank,flag_win);

printf("Rank %d: final flag_array[0] = %d\n", rank, flag_array[0]);
sleep(1);

    MPI_Test(&dummy_req,&dummy_flag, MPI_STATUS_IGNORE);
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE,rank,0, flag_win);
 }
      
 //MPI_Win_unlock_all(flag_win);
 
 printf("Rank %d: final flag_array[0] = %d\n", rank, flag_array[0]);

//MPI_Win_unlock( rank, flag_win);
MPI_Win_unlock(rank,flag_win);
    MPI_Win_free(&flag_win);
    MPI_Free_mem(flag_array);   
    
// printf("This is the end");
 
return MPI_SUCCESS;
}



int alltoallv_init(const void* sendbuf,
       const int sendcounts[],
       const int sdispls[],
       MPI_Datatype sendtype,
       void* recvbuf,
       const int recvcounts[],
       const int rdispls[],
       MPI_Datatype recvtype,
       MPIX_Comm* xcomm,
       MPIX_Info* xinfo,
       MPIX_Request** request_ptr)
{   
    int rank, num_procs;
    int err;

    err = MPI_Comm_rank(xcomm->global_comm, &rank);
    
    err = MPI_Comm_size(xcomm->global_comm, &num_procs);
    
    MPIX_Request* request;
    err = MPIX_Request_init(&request);
    

    request->global_n_msgs = 2 * num_procs;
    err = allocate_requests(request->global_n_msgs, &(request->global_requests));
    

    int tag = 102944;
    int send_proc, recv_proc;
    int send_pos, recv_pos;
    MPI_Status status;

    int send_size, recv_size;
    err = MPI_Type_size(sendtype, &send_size);
    

    err = MPI_Type_size(recvtype, &recv_size);
    
    char* send_buffer = (char*)(sendbuf);
    char* recv_buffer = (char*)(recvbuf);

  
    // Initialize persistent send and receive requests
    for (int i = 0; i < num_procs; i++) {
        send_proc = (rank + i) % num_procs;
        recv_proc = (rank - i + num_procs) % num_procs;

        send_pos = sdispls[send_proc] * send_size;
        recv_pos = rdispls[recv_proc] * recv_size;

       
        // Initialize persistent send request
        err = MPI_Send_init(send_buffer + send_pos, sendcounts[send_proc], sendtype, send_proc, tag,
                            xcomm->global_comm, &(request->global_requests[2 * i]));
       
        // Initialize persistent receive request
        err = MPI_Recv_init(recv_buffer + recv_pos, recvcounts[recv_proc], recvtype, recv_proc, tag,
                            xcomm->global_comm, &(request->global_requests[2 * i + 1]));
       
    }

    // Start the communication
    err = MPI_Startall(request->global_n_msgs, request->global_requests);
    

    // Wait for all communications to finish
    err = MPI_Waitall(request->global_n_msgs, request->global_requests, MPI_STATUSES_IGNORE);
   

    // Set the request pointer to the newly created request array
    *request_ptr = request;

    return MPI_SUCCESS; 
}







int alltoallv_nonblocking_init(const void* sendbuf,
       const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
       const int recvcounts[],
       const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* xcomm,
        MPIX_Info* xinfo,
        MPIX_Request** request_ptr)
{

int num_procs;
    MPI_Comm_size(xcomm->global_comm, &num_procs);

alltoallv_init(sendbuf,sendcounts,sdispls, sendtype, recvbuf, recvcounts,rdispls, recvtype, xcomm,
            xinfo, request_ptr);
    (*request_ptr)->batch = num_procs;

    return MPI_SUCCESS;


}



int alltoallv_pairwise_init(const void* sendbuf,
       const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
       const int recvcounts[],
       const int rdispls[],
        MPI_Datatype recvtype,
        //MPI_Comm comm,
        MPIX_Comm* xcomm,
        MPIX_Info* xinfo,
        MPIX_Request** request_ptr)
{

alltoallv_init(sendbuf,sendcounts,sdispls, sendtype, recvbuf, recvcounts,rdispls, recvtype, xcomm,
            xinfo, request_ptr);
    (*request_ptr)->batch = 1;

    return MPI_SUCCESS;



}



int alltoallv_pairwise_nonblocking(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPI_Comm comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    // Tuning Parameter : number of non-blocking messages between waits 
    int nb_stride = 5;

    int tag = 103044;
    int ctr;
    int send_proc, recv_proc;
    int send_pos, recv_pos;
    MPI_Status status;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    MPI_Request* requests = (MPI_Request*)malloc(2*nb_stride*sizeof(MPI_Request));

    memcpy(
        recvbuf + (rdispls[rank] * recv_size),
        sendbuf + (sdispls[rank] * send_size), 
        sendcounts[rank] * send_size);        

    // For each step i
    // exchange among procs stride (i+1) apart
    ctr = 0;
    for (int i = 1; i < num_procs; i++)
    {
        send_proc = rank + i;
        if (send_proc >= num_procs)
            send_proc -= num_procs;
        recv_proc = rank - i;
        if (recv_proc < 0)
            recv_proc += num_procs;

        send_pos = sdispls[send_proc] * send_size;
        recv_pos = rdispls[recv_proc] * recv_size;

        MPI_Isend(sendbuf + send_pos, sendcounts[send_proc], sendtype, send_proc, tag,
                comm, &(requests[ctr++]));
        MPI_Irecv(recvbuf + recv_pos, recvcounts[recv_proc], recvtype, recv_proc, tag,
                comm, &(requests[ctr++]));

        if (i % nb_stride == 0)
        {
            MPI_Waitall(2*nb_stride, requests, MPI_STATUSES_IGNORE);
            ctr = 0;
        }
    }
    
    if (ctr)
        MPI_Waitall(ctr, requests, MPI_STATUSES_IGNORE);

    free(requests);

    return 0;
}

int alltoallv_waitany(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPI_Comm comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    // Tuning Parameter : number of non-blocking messages between waits 
    int nb_stride = 5;

    int tag = 103044;
    int ctr;
    int send_proc, recv_proc;
    int send_pos, recv_pos;
    MPI_Status status;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    MPI_Request* requests = (MPI_Request*)malloc(2*nb_stride*sizeof(MPI_Request));

    memcpy(
        recvbuf + (rdispls[rank] * recv_size),
        sendbuf + (sdispls[rank] * send_size), 
        sendcounts[rank] * send_size);        

    // For each step i
    // exchange among procs stride (i+1) apart
    ctr = 0;
    for (int i = 1; i <= nb_stride && i < num_procs; i++)
    {
        send_proc = rank + i;
        if (send_proc >= num_procs)
            send_proc -= num_procs;
        recv_proc = rank - i;
        if (recv_proc < 0)
            recv_proc += num_procs;

        send_pos = sdispls[send_proc] * send_size;
        recv_pos = rdispls[recv_proc] * recv_size;

        MPI_Isend(sendbuf + send_pos, sendcounts[send_proc], sendtype, send_proc, tag,
                comm, &(requests[ctr++]));
        MPI_Irecv(recvbuf + recv_pos, recvcounts[recv_proc], recvtype, recv_proc, tag,
                comm, &(requests[ctr++]));

    }

    if (nb_stride >= num_procs)
    {
        MPI_Waitall(2*(num_procs-1), requests, MPI_STATUSES_IGNORE);
        free(requests);
        return 0;
    }

    int send_idx = nb_stride;
    int recv_idx = nb_stride;
    int idx;
    while (1)
    {
        MPI_Waitany(2*nb_stride, requests, &idx, MPI_STATUSES_IGNORE);

        if (idx == MPI_UNDEFINED)
        {
            break;
        }

        if (idx % 2 == 0 && send_idx < num_procs)
        {
            send_proc = rank + send_idx;
            if (send_proc >= num_procs)
                send_proc -= num_procs;
            send_pos = sdispls[send_proc] * send_size;
            MPI_Isend(sendbuf + send_pos, sendcounts[send_proc], sendtype, send_proc, tag,
                    comm, &(requests[idx]));
            send_idx++;
        }
        else if (idx % 2 == 1 && recv_idx < num_procs)
        {
            recv_proc = rank - recv_idx;
            if (recv_proc < 0)
                recv_proc += num_procs;
            recv_pos = rdispls[recv_proc] * recv_size;

            MPI_Irecv(recvbuf + recv_pos, recvcounts[recv_proc], recvtype, recv_proc, tag,
                    comm, &(requests[idx]));
            recv_idx++;
        }
    }

    free(requests);

    return 0;
}

// 2-Step Aggregation (large messages)
// Gather all data to be communicated between nodes
// Send to node+i, recv from node-i
// TODO (For Evelyn to look at sometime?) : 
//     What is the best way to aggregate very large messages?
//     Should we load balance to make sure all processes per node
//         send equal amount of data? (ideally, yes)
//     Should we use S. Lockhart's  'ideal' aggregation, setting
//         a tolerance.  Any message with size < tolerance, aggregate
//         this data with other processes locally.
//     How should we aggregate data when using GPU memory??
int alltoallv_pairwise_loc(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* mpi_comm)
{
    int rank, num_procs;
    int local_rank, PPN; 
    int num_nodes, rank_node;
    MPI_Comm_rank(mpi_comm->global_comm, &rank);
    MPI_Comm_size(mpi_comm->global_comm, &num_procs);
    MPI_Comm_rank(mpi_comm->local_comm, &local_rank);
    MPI_Comm_size(mpi_comm->local_comm, &PPN);
    num_nodes = mpi_comm->num_nodes;
    rank_node = mpi_comm->rank_node;

    const char* send_buffer = (char*) sendbuf;
    char* recv_buffer = (char*) recvbuf;
    int sbytes, rbytes;
    MPI_Type_size(sendtype, &sbytes);
    MPI_Type_size(recvtype, &rbytes);

    int tag = 102913;
    int send_proc, recv_proc;
    int send_pos, recv_pos;
    int send_node, recv_node;
    MPI_Status status;

    int final_recvcount = 0;
    for (int i = 0; i < num_procs; i++)
        final_recvcount += recvcounts[i];

    /************************************************
     * Step 1 : Send aggregated data to node
     ***********************************************/
    int sendcount, recvcount;
    int* global_recvcounts = (int*)malloc(num_procs*sizeof(int));
    int global_recvcount = 0;
    // Send to node + i
    // Recv from node - i
    for (int i = 0; i < num_nodes; i++)
    {
        send_node = rank_node + i;
        if (send_node >= num_nodes)
            send_node -= num_nodes;
        recv_node = rank_node - i;
        if (recv_node < 0)
            recv_node += num_nodes;

        MPI_Sendrecv(&(sendcounts[send_node*PPN]), PPN, MPI_INT,
                send_node*PPN+local_rank, tag,
                &(global_recvcounts[recv_node*PPN]), PPN, MPI_INT,
                recv_node*PPN+local_rank, tag,
                mpi_comm->global_comm, &status); 
    }

    int maxrecvcount = final_recvcount;
    if (global_recvcount > maxrecvcount)
        maxrecvcount = global_recvcount;
    char* tmpbuf = (char*)malloc(maxrecvcount*rbytes);
    char* contigbuf = (char*)malloc(maxrecvcount*rbytes);

    // Send to node + i
    // Recv from node - i
    for (int i = 0; i < num_nodes; i++)
    {
        send_node = rank_node + i;
        if (send_node >= num_nodes)
            send_node -= num_nodes;
        recv_node = rank_node - i;
        if (recv_node < 0)
            recv_node += num_nodes;
        send_pos = sdispls[send_node * PPN];
        recv_pos = rdispls[recv_node * PPN];

        sendcount = 0;
        recvcount = 0;
        for (int j = 0; j < PPN; j++)
        {
            sendcount += sendcounts[send_node*PPN+j];
            recvcount += global_recvcounts[recv_node*PPN+j];
        }

        MPI_Sendrecv(sendbuf + send_pos*sbytes, sendcount, 
                sendtype, send_node*PPN + local_rank, tag,
                tmpbuf + recv_pos*rbytes, recvcount, 
                recvtype, recv_node*PPN + local_rank, tag, 
                mpi_comm->global_comm, &status);
    }

    /************************************************
     * Step 2 : Redistribute received data within node
     ************************************************/
    int* ppn_ctr = (int*)malloc(PPN*sizeof(int));
    int* ppn_displs = (int*)malloc((PPN+1)*sizeof(int));
    for (int i = 0; i < PPN; i++)
        ppn_ctr[i] = 0;
    for (int i = 0; i < num_nodes; i++)
        for (int j = 0; j < PPN; j++)
            ppn_ctr[j] += global_recvcounts[i*PPN+j];
    ppn_displs[0] = 0;
    for (int i = 0; i < PPN; i++)
    {
        ppn_displs[i+1] = ppn_displs[i] + ppn_ctr[i];
        ppn_ctr[i] = 0;
    }

    // TODO (for Evelyn to look into?) : 
    //     Currently, re-pack data here
    //     We recv'd data from each node
    //     Now we re-pack it so that it is
    //     ordered by destination process rather
    //     than source node.
    //     Packing can be expensive! Should we
    //     use MPI Datatypes?  Or send num_nodes 
    //     different messages to each of the PPN
    //     local processes?

    int ctr = 0;
    recvcount = 0;
    for (int i = 0; i < num_nodes; i++)
        for (int j = 0; j < PPN; j++)
        {
            recvcount = global_recvcounts[i*PPN+j];
            memcpy(recvbuf + (ppn_displs[j] + ppn_ctr[j])*rbytes,
                    tmpbuf + ctr*rbytes,
                    recvcount*rbytes);
            ctr += recvcount;
            ppn_ctr[j] += recvcount;
        }

    // Send to local_rank + i
    // Recv from local_rank + i
    ctr = 0;
    for (int i = 0; i < PPN; i++)
    {
        send_proc = local_rank + i;
        if (send_proc >= PPN)
            send_proc -= PPN;
        recv_proc = local_rank - i;
        if (recv_proc < 0)
            recv_proc += PPN;

        send_pos = ppn_displs[send_proc] * rbytes;
        recvcount = 0;
        for (int j = 0; j < num_nodes; j++)
            recvcount += recvcounts[j*PPN+i];

        MPI_Sendrecv(recvbuf + send_pos, ppn_ctr[send_proc], recvtype,
                send_proc, tag,
                tmpbuf + ctr*rbytes, recvcount, recvtype,
                recv_proc, tag,
                mpi_comm->local_comm, &status);

        ppn_ctr[recv_proc] = ctr;

        ctr += recvcount;
    }

    for (int i = 0; i < PPN; i++)
    {
        for (int j = 0; j < num_nodes; j++)
        {
            memcpy(recvbuf + rdispls[j*PPN+i]*rbytes,
                    tmpbuf + ppn_ctr[i]*rbytes,
                    recvcounts[j*PPN+i]*rbytes);
            ppn_ctr[i] += recvcounts[j*PPN+i];
        }
    }

    free(ppn_ctr);
    free(ppn_displs);
    free(global_recvcounts);
    free(contigbuf);
    free(tmpbuf);

    return 0;
}

