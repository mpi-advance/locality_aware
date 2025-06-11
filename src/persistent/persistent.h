#ifndef MPI_ADVANCE_PERSISTENT_H
#define MPI_ADVANCE_PERSISTENT_H

#include "communicator/locality_comm.h"
#include "communicator/mpix_comm.h"
#include "utils/utils.h"

#ifdef __cplusplus
extern "C"
{
#endif

struct _MPIX_Request; // forward declaration
typedef struct _MPIX_Request MPIX_Request;

struct _MPIX_Request
{
    // Message counts
    // Will only use global unless locality-aware
    int local_L_n_msgs;
    int local_S_n_msgs;
    int local_R_n_msgs;
    int global_n_msgs;

    // MPI Request arrays
    // Will only use global unless locality-aware
    MPI_Request* local_L_requests;
    MPI_Request* local_S_requests;
    MPI_Request* local_R_requests;
    MPI_Request* global_requests;

    // Pointer to locality communication, only for locality-aware
    LocalityComm* locality;

    // Pointer to sendbuf and recvbuf
    const void* sendbuf; // pointer to sendbuf (where original data begins)
    void* recvbuf; // pointer to recvbuf (where final data goes)

    // Number of bytes per receive object (for locality-aware)
    int recv_size;

    // Block size : for strided/blocked communication
    int block_size;

    int tag;
    int reorder;

    // For allocating cpu buffers for heterogeneous communication
#ifdef GPU
    void* cpu_sendbuf; // for copy-to-cpu
    void* cpu_recvbuf; // for copy-to-cpu
#endif

    // Keep track of which start/wait functions to call for given request
    int (*start_function)(MPIX_Request* request);
    int (*wait_function)(MPIX_Request* request, MPI_Status* status);
};

typedef int (*mpix_start_ftn)(MPIX_Request* request);
typedef int (*mpix_wait_ftn)(MPIX_Request* request, MPI_Status* status);

// Starting locality-aware requests
// 1. Start Local_L
// 2. Start and wait for local_S
// 3. Start global
int MPIX_Start(MPIX_Request* request);


// Wait for locality-aware requests
// 1. Wait for global
// 2. Start and wait for local_R
// 3. Wait for local_L
int MPIX_Wait(MPIX_Request* request, MPI_Status* status);

int MPIX_Request_free(MPIX_Request** request);

void init_request(MPIX_Request** request_ptr);
void allocate_requests(int n_requests, MPI_Request** request_ptr);
void destroy_request(MPIX_Request* request);


    
#ifdef __cplusplus
}
#endif


#endif
