#ifndef MPI_ADVANCE_UTILS_H
#define MPI_ADVANCE_UTILS_H

#ifdef HIP
#include "utils_hip.h"
#endif

#ifdef CUDA
#include "utils_cuda.h"
#endif

#ifdef __cplusplus
extern "C"
{
#endif

void sort(int n_objects, int* object_indices, int* object_values);
void rotate(void* ref, int new_start_byte, int end_byte);
void reverse(void* recvbuf, int n_bytes, int var_bytes);

#ifdef __cplusplus
}
#endif


#endif
