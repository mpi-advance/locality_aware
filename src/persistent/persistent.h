#ifndef MPI_ADVANCE_PERSISTENT_H
#define MPI_ADVANCE_PERSISTENT_H

#include "locality/locality_comm.h"

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

    void* start_function;
    void* wait_function; 

} MPIX_Request;

typedef int (*mpix_start_ftn)(MPIX_Request* request);
typedef int (*mpix_wait_ftn)(MPIX_Request* request, MPI_Status* status);

int MPIX_Start(MPIX_Request* request);
int MPIX_Wait(MPIX_Request* request, MPI_Status* status);

int neighbor_start(MPIX_Request* request);
int neighbor_wait(MPIX_Request* request, MPI_Status* status);



int MPIX_Request_free(MPIX_Request* request);

#ifdef __cplusplus
}
#endif


#endif
