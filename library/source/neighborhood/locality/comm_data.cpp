#include "neighborhood/comm_data.h"
#include "stdlib.h"

void init_num_msgs(CommData* data, int num_msgs)
{
    data->num_msgs = num_msgs;
    if (data->num_msgs)
    {
        data->procs = (int*)malloc(sizeof(int) * data->num_msgs);
        data->counts = (int*)malloc(sizeof(int) * data->num_msgs);
    }
    data->indptr    = (int*)malloc(sizeof(int) * (data->num_msgs + 1));
    data->indptr[0] = 0;
}

void init_size_msgs(CommData* data, int size_msgs)
{
    data->size_msgs = size_msgs;
    if (data->size_msgs)
    {
        data->indices = (int*)malloc(data->size_msgs * sizeof(int));
    }
}

void destroy_comm_data(CommData* data)
{
    if (data->procs != NULL)
    {
        free(data->procs);
        data->procs = NULL;
    }
    if (data->indptr != NULL)
    {
        free(data->indptr);
        data->indptr = NULL;
    }
    if (data->counts != NULL)
    {
        free(data->counts);
        data->counts = NULL;
    }
    if (data->indices != NULL)
    {
        free(data->indices);
        data->indices = NULL;
    }
    if (data->buffer != NULL)
    {
        free(data->buffer);
        data->buffer = NULL;
    }

    free(data);
}
