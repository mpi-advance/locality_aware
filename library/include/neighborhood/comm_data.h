#ifndef COMM_DATA_H
#define COMM_DATA_H

typedef struct _CommData
{
    int num_msgs;
    int size_msgs;
    int datatype_size;
    int* procs;
    int* indptr;
    int* counts;
    int* indices;
    char* buffer;
} CommData;

void destroy_comm_data(CommData* data);
/** @brief Sets the CommData::num_msgs to provided value */
void init_num_msgs(CommData* data, int num_msgs);
/** @brief Sets CommData::size_msgs and allocates CommData::indices for indexing messages **/
void init_size_msgs(CommData* data, int size_msgs);

#endif
