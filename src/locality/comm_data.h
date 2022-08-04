#ifndef MPI_ADVANCE_COMM_DATA_H
#define MPI_ADVANCE_COMM_DATA_H

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _CommData
{
    int num_msgs;
    int size_msgs;
    int datatype_size;
    int* procs;
    int* indptr;
    int* indices;
    char* buffer;
} CommData;

void init_comm_data(CommData** comm_data_ptr, MPI_Datatype datatype);
void destroy_comm_data(CommData* data);
void init_num_msgs(CommData* data, int num_msgs);
void init_size_msgs(CommData* data, int size_msgs);
void finalize_comm_data(CommData* data);

#ifdef __cplusplus
}
#endif


#endif
