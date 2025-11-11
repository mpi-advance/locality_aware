#ifndef MPIL_REQUEST_H
#define MPIL_REQUEST_H

#include <mpi.h>

#include "communicator/locality_comm.h"

#ifdef __cplusplus
extern "C" {
#endif

/** 
   @brief replacement for MPI_Request for use with library API. 
   @details 
	 class is protected, limited direct access to class members through API calls.
     initialized by init_request
	 freed by destroy_request
	 Contains multiple requests and buffers to manage complex communication. 
	 Contains function pointer to appropriate start and wait functions. 
*/

typedef struct _MPIL_Request
{
    // Message counts
    // Will only use global unless locality-aware
	/** @brief intra-node message count **/
    int local_L_n_msgs;  
	/** @brief sent message count **/
    int local_S_n_msgs;
	/** @brief received message count **/
    int local_R_n_msgs;
	/** @brief number of inter-node messages **/
    int global_n_msgs;

    // MPI Request arrays
    // Will only use global unless locality-aware
	/** @brief requests to manage of intra-node messages **/
    MPI_Request* local_L_requests;
	/** @brief requests to control sent messages **/	
    MPI_Request* local_S_requests;
	/** @brief requests to control recieved messages **/	
    MPI_Request* local_R_requests;
	/** @brief requests to manage of inter-node messages **/
    MPI_Request* global_requests;

    // Pointer to locality communication, only for locality-aware
	/** @brief pointer to locality communication information if locality_aware **/
    LocalityComm* locality;

    // Pointer to sendbuf and recvbuf
    const void* sendbuf;  // pointer to sendbuf (where original data begins)
	void* recvbuf;        // pointer to recvbuf (where final data goes)

	/** @brief number of bytes per receive object, locality-aware only **/
    int recv_size;

    /** @briefBlock size : for strided/blocked communication **/
	int block_size;

    int tag;
	
    int reorder;

#ifdef GPU
	/** @brief allocate cpu buffers for copy-to-cpu algorithms **/
    void* cpu_sendbuf; 
	/** @brief allocate cpu buffers for copy-to-cpu algorithms **/
    void* cpu_recvbuf; 
#endif
	/** @brief function pointer to MPIL_Start or MPIL_neighbor_start **/
    int (*start_function)(struct _MPIL_Request* request);
	/** @brief function pointer to MPIL_Wait or MPIL_neighbor_wait **/
    int (*wait_function)(struct _MPIL_Request* request, MPI_Status* status);
} MPIL_Request;

void init_request(MPIL_Request** request_ptr);

/** @brief allocate enough space for n MPI_Requests 
	@param [in] n_request how many requests need space
	@param [out] request_ptr pointer to start of allocated memory
**/
void allocate_requests(int n_requests, MPI_Request** request_ptr);
void destroy_request(MPIL_Request* request);

#ifdef __cplusplus
}
#endif

#endif