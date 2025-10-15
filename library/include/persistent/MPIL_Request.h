#ifndef MPIL_REQUEST_H
#define MPIL_REQUEST_H

#include <mpi.h>
#include "communicator/locality_comm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _MPIL_Request
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
    const void* sendbuf;  // pointer to sendbuf (where original data begins)
    void* recvbuf;        // pointer to recvbuf (where final data goes)

    // Number of bytes per receive object (for locality-aware)
    int recv_size;

    // Block size : for strided/blocked communication
    int block_size;

    int tag;
    int reorder;

    // For allocating cpu buffers for heterogeneous communication
#ifdef GPU
    void* cpu_sendbuf;  // for copy-to-cpu
    void* cpu_recvbuf;  // for copy-to-cpu
#endif

    // Keep track of which start/wait functions to call for given request
    int (*start_function)(MPIL_Request* request);
    int (*wait_function)(MPIL_Request* request, MPI_Status* status);
} MPIL_Request; 

void init_request(MPIL_Request** request_ptr);
void allocate_requests(int n_requests, MPI_Request** request_ptr);
void destroy_request(MPIL_Request* request);

#ifdef __cplusplus
}
#endif

#endif