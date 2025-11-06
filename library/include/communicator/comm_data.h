#ifndef MPI_ADVANCE_COMM_DATA_H
#define MPI_ADVANCE_COMM_DATA_H

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif
/** @brief structure containing meta data about a message**/
typedef struct _CommData
{
	/** @brief number of message sent between sender and receiver **/
    int num_msgs;
	/** @brief size of the message sent between sender and receiver **/
    int size_msgs;
	/** @brief size of the  **/
    int datatype_size;
	/** @brief processes involved in the communication **/
    int* procs;
	/** @brief pointer to index  **/
    int* indptr;
	/** @brief indexes for message  **/
    int* indices;
	/** @brief buffer containing copy of message  **/
    char* buffer;
} CommData;

/**@brief allocates commData with handle at comm_data_ptr, sets datatype_size to size of supplied datatype**/
void init_comm_data(CommData** comm_data_ptr, MPI_Datatype datatype);
/** @brief delete comm and free allocated memory **/
void destroy_comm_data(CommData* data);

void init_num_msgs(CommData* data, int num_msgs);
/** @brief allocate enough space for indexing messages (int[num_msg) **/
void init_size_msgs(CommData* data, int size_msgs);

/** @brief allocates buffer 
 *  @details 
 *	mallocs enough space for buffer
**/
void finalize_comm_data(CommData* data);

#ifdef __cplusplus
}
#endif

#endif
