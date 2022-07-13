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

void init_comm_data(CommData** comm_data_ptr)
{
    CommData* data = (CommData*)malloc(sizeof(CommData));

    data->num_msgs = 0;
    data->size_msgs = 0;
    data->procs = NULL;
    data->indptr = NULL;
    data->indices = NULL;
    data->buffer = NULL;

    *comm_data_ptr = data;
}

// TODO : delete[]??
void destroy_comm_data(CommData* data)
{
    if (data->procs) free(data->procs);
    if (data->indptr) free(data->indptr);
    if (data->indices) free(data->indices);
    if (data->buffer) free(data->buffer);

    free(data);
}


void init_num_msgs(CommData* data, int num_msgs)
{
    data->num_msgs = num_msgs;
    if (data->num_msgs)
        data->procs = (int*)malloc(sizeof(int)*data->num_msgs);
    data->indptr = (int*)malloc(sizeof(int)*(data->num_msgs+1));
    data->indptr[0] = 0;
}

void init_size_msgs(CommData* data, int size_msgs)
{
    data->size_msgs = size_msgs;
    if (data->size_msgs)
    {
        data->indices = (int*)malloc(sizeof(int)*data->size_msgs);
    }
}




#endif
