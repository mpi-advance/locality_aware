#ifndef MPI_ADVANCE_PERSISTENT_H
#define MPI_ADVANCE_PERSISTENT_H

#include "locality/locality_comm.h"
#include <string.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _MPIX_Request
{
    int local_L_n_msgs;
    int local_S_n_msgs;
    int local_R_n_msgs;
    int global_n_msgs;

    MPI_Request* local_L_requests;
    MPI_Request* local_S_requests;
    MPI_Request* local_R_requests;
    MPI_Request* global_requests;

    LocalityComm* locality;

    const void* sendbuf; // pointer to sendbuf (where original data begins)
    void* recvbuf; // pointer to recvbuf (where final data goes)
    int recv_size;
    int batch;

    void* start_function; // points to MPIX_Start impl to call
    void* wait_function;  // points to MPIX_Wait impl to call

    MPIX_Comm* xcomm; // always just reference to, not allocated here
    int* sdispls;
    int* put_displs;
    int* send_sizes;
    int* recv_sizes;
    int n_puts;


} MPIX_Request;

typedef int (*mpix_start_ftn)(MPIX_Request* request);
typedef int (*mpix_wait_ftn)(MPIX_Request* request, MPI_Status* status);

int MPIX_Start(MPIX_Request* request);
int MPIX_Wait(MPIX_Request* request, MPI_Status* status);

int neighbor_start(MPIX_Request* request);
int neighbor_wait(MPIX_Request* request, MPI_Status* status);

int batch_start(MPIX_Request* request);
int batch_wait(MPIX_Request* request, MPI_Status* status);
int rma_start(MPIX_Request* request);
int rma_wait(MPIX_Request* request, MPI_Status* status);
int rma_lock_start(MPIX_Request* request);
int rma_lock_wait(MPIX_Request* request, MPI_Status* status);



int MPIX_Request_init(MPIX_Request** request);
int MPIX_Request_free(MPIX_Request* request);

int allocate_requests(int n, MPI_Request** request_ptr);

#ifdef __cplusplus
}
#endif


#endif
