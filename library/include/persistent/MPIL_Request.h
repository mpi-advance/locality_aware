#ifndef MPIL_REQUEST_H
#define MPIL_REQUEST_H

#include <mpi.h>

#include "utils/MPIL_Alloc.h"

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

    /** @brief Pointer to the user's original send buffer*/
    const void* sendbuf;
    /** @brief Pointer to the user's original receive buffer */
    void* recvbuf;

    /** @brief Pointer to new send buffer for data to be 
     * packed into at intermediate steps */
    void* tmp_sendbuf;
    /** @brief Pointer to new recv buffer for data to be 
     * recvd into at intermediate steps */
    void* tmp_recvbuf;
    /** @brief indices of input buffer to be packed **/
    int* send_indices;
    /** @brief indices of received buffer to be unpacked **/
    int* recv_indices;
    /** @brief size of sendbuf/send_indices **/ 
    int size_sends;
    /** @brief size of recvbuf/recv_indices **/ 
    int size_recvs;
    /** @brief size of sendtype **/
    int send_size;
    /** @brief size of recvtype **/

    // Pointers to MPI_Requests for aggregated communication
    /** @brief Fully local communication **/
    MPIL_Request* local_L_request;
    /** @brief Initial local aggregation **/
    MPIL_Request* local_S_request;
    /** @brief Final local disaggrgation **/
    MPIL_Request* local_R_request;

    /** @brief Number of bytes per receive object, locality-aware only **/
    int recv_size;
    /** @brief Block size for strided/blocked communication **/
    int block_size;

    MPI_Comm global_comm;
    MPI_Comm local_comm;

    /** @brief Flag for if we want MPIL to reorder requests based on order of arrival
     * during first iteration **/
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
    /** @brief Allocated cpu-based send buffers for copy-to-cpu algorithms 
     *         CPU buffers are stored in sendbuf, original sendbuf is
     *         stored here */
    const void* gpu_sendbuf;
    /** @brief Allocated cpu-based receive buffers for copy-to-cpu algorithms
     *         CPU buffers are stored in recvbuf, original recvbuf is
     *         stored here */
    void* gpu_recvbuf;
    /** @brief points to sendbuf, but not const, for copy-to-CPU */
    void* tmp_gpubuf;
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
