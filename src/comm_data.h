#ifndef MPI_ADVANCE_COMM_DATA_H
#define MPI_ADVANCE_COMM_DATA_H

#include <stdlib.h>
#include <stdio.h>

typedef struct _CommData
{
    int num_msgs;
    int size_msgs;
    int* procs;
    int* indptr;
    int* indices;
    char* buffer;
} CommData;

void init_comm_data(CommData** comm_data_ptr);
void destroy_comm_data(CommData* data);
void init_num_msgs(CommData* data, int num_msgs);
void init_size_msgs(CommData* data, int size_msgs);




#endif
