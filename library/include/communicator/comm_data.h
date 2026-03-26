#ifndef MPI_ADVANCE_COMM_DATA_H
#define MPI_ADVANCE_COMM_DATA_H

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif
/** @brief Structure containing metadata about a message**/
typedef struct _CommData
{
    /** @brief Number of messages sent between sender and receiver **/
    int num_msgs;
    /** @brief size of the message sent between sender and receiver **/
    int size_msgs;
    /** @brief Size of the datatype used in the communication **/
    int datatype_size;
    /** @brief Number of processes involved in the communication **/
    int* procs;
    /** @brief pointer to index  **/
    int* indptr;
    /** @brief indexes for message  **/
    int* indices;
    /** @brief buffer containing copy of message  **/
    char* buffer;
} CommData;

/** @brief ::CommData constructor that sets all values to 0, except for CommData::datatype_size **/
void init_comm_data(CommData** comm_data_ptr, MPI_Datatype datatype);
/** @brief ::CommData destructor that frees allocated memory **/
void destroy_comm_data(CommData* data);
/** @brief Sets the CommData::num_msgs to provided value */
void init_num_msgs(CommData* data, int num_msgs);
/** @brief Sets CommData::size_msgs and allocates CommData::indices for indexing messages **/
void init_size_msgs(CommData* data, int size_msgs);
/** @brief Allocates CommData::buffer to the size of `(CommData::size_msgs * CommData::datatype_size)` bytes **/
void finalize_comm_data(CommData* data);

#ifdef __cplusplus
}
#endif

#endif
