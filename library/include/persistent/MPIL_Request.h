#ifndef MPIL_REQUEST_H
#define MPIL_REQUEST_H

#include <mpi.h>

#include "communicator/locality_comm.h"
#include "utils/MPIL_Alloc.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief A custom MPI_Request struct used for the library's persistent collectives
 * @details For external users, there is limited direct access to class members through
 * API calls. Contains multiple requests and buffers to manage complex communication.
 * Contains function pointer to appropriate start and wait functions.
 */
typedef struct _MPIL_Request
{
    // Message counts; Will only use global unless locality-aware
    /** @brief Intra-node message count **/
    int local_L_n_msgs;
    /** @brief Sent message count **/
    int local_S_n_msgs;
    /** @brief Received message count **/
    int local_R_n_msgs;
    /** @brief Number of inter-node messages **/
    int global_n_msgs;

    // MPI Request arrays; Will only use global unless locality-aware
    /** @brief Requests to manage of intra-node messages **/
    MPI_Request* local_L_requests;
    /** @brief Requests to control sent messages **/
    MPI_Request* local_S_requests;
    /** @brief Requests to control received messages **/
    MPI_Request* local_R_requests;
    /** @brief Requests to manage of inter-node messages **/
    MPI_Request* global_requests;

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

    MPI_Comm global_comm;
    MPI_Comm local_comm;

    int reorder;

    void* tmpbuf;
    MPIL_Free_ftn free_ftn;

    // Only needed for allreduce (MPI_Reduce_local call)
    /** @brief count Number of datatypes to reduce in MPI_Reduce_local **/
    int count;
    /** @brief datatype MPI_Datatype to be reduced in MPI_Reduce_local **/
    MPI_Datatype datatype;
    /** @brief op MPI_Op to be used in MPI_Reduce_local **/
    MPI_Op op;
    /** @brief num_ops int number of local operations **/
    int num_ops;
    

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
} MPIL_Request;

/** @brief Constructor for ::MPIL_Request. Initializes most members to 0. */
void init_request(MPIL_Request** request_ptr);

/** @brief Allocate enough space for n MPI_Requests
        @param [in] n_request how many requests need space
        @param [out] request_ptr pointer to start of allocated memory
**/
void allocate_requests(int n_requests, MPI_Request** request_ptr);

#ifdef __cplusplus
}
#endif

#endif
