#ifndef MPI_ADVANCE_PERSISTENT_H
#define MPI_ADVANCE_PERSISTENT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MPIL_Request MPIL_Request; // forward declaration

typedef int (*mpix_start_ftn)(MPIL_Request* request);
typedef int (*mpix_wait_ftn)(MPIL_Request* request, MPI_Status* status);

// Starting locality-aware requests
// 1. Start Local_L
// 2. Start and wait for local_S
// 3. Start global
int MPIL_Start(MPIL_Request* request);

// Wait for locality-aware requests
// 1. Wait for global
// 2. Start and wait for local_R
// 3. Wait for local_L
int MPIL_Wait(MPIL_Request* request, MPI_Status* status);

int MPIL_Request_free(MPIL_Request** request);

#ifdef __cplusplus
}
#endif

#endif
