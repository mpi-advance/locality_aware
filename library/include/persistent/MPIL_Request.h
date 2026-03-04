#ifndef MPIL_REQUEST_H
#define MPIL_REQUEST_H

#include <mpi.h>

#include "communicator/locality_comm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _MPIL_Request MPIL_Request;
/** @brief A custom MPI_Request struct used for the library's persistent collectives
 * @details For external users, there is limited direct access to class members through
 * API calls. Contains multiple requests and buffers to manage complex communication.
 * Contains function pointer to appropriate start and wait functions.
 */
struct _MPIL_Request
{
    /** @brief Number of messages **/
    int n_msgs;

    /** @brief array of MPI Requests **/
    MPI_Request* requests;


    // Pointers to MPI_Requests for aggregated communication
    /** @brief Fully local communication **/
    MPIL_Request* local_L_request;
    /** @brief Initial local aggregation **/
    MPIL_Request* local_S_request;
    /** @brief Final local disaggrgation **/
    MPIL_Request* local_R_request;

    /** @brief Pointer to locality communication information if using locality-aware
     * variants **/
    LocalityComm* locality;

    /** @brief Pointers to the user's original send buffer */
    const void* sendbuf;
    /** @brief Pointer to the user's original receive buffer */
    void* recvbuf;

    /** @brief Number of bytes per receive object, locality-aware only **/
    int recv_size;
    /** @brief Block size for strided/blocked communication **/
    int block_size;

    int reorder;

#ifdef GPU
    /** @brief Allocated cpu-based send buffers for copy-to-cpu algorithms **/
    void* cpu_sendbuf;
    /** @brief Allocated cpu-based receive buffers for copy-to-cpu algorithms **/
    void* cpu_recvbuf;
#endif
    /** @brief Function pointer to MPIL_Start or MPIL_neighbor_start **/
    int (*start_function)(struct _MPIL_Request* request);
    /** @brief Function pointer to MPIL_Wait or MPIL_neighbor_wait **/
    int (*wait_function)(struct _MPIL_Request* request, MPI_Status* status);
};

/** @brief Constructor for ::MPIL_Request. Initializes most members to 0. */
void init_request(MPIL_Request** request_ptr);

/** @brief Allocate enough space for n MPI_Requests
        @param [in] n_request how many requests need space
        @param [out] request_ptr pointer to start of allocated memory
**/
void allocate_requests(int n_requests, MPIL_Request* request);

#ifdef __cplusplus
}
#endif

#endif
