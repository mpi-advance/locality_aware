#ifndef MPI_ADVANCE_UTILS_H
#define MPI_ADVANCE_UTILS_H

#include "mpi.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _MPIX_Info
{
    int tag;
    int crs_num_initialized;
    int crs_size_initialized;
} MPIX_Info;

int MPIX_Info_init(MPIX_Info** info);
int MPIX_Info_free(MPIX_Info** info);

void sort(int n_objects, int* object_indices, int* object_values);
void rotate(void* ref, int new_start_byte, int end_byte);
void reverse(void* recvbuf, int n_bytes, int var_bytes);

#ifdef __cplusplus
}
#endif


#endif
