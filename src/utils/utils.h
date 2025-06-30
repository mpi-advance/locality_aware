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


// MPIX Info Object
typedef struct _MPIX_Info
{
    int crs_num_initialized;
    int crs_size_initialized;
} MPIX_Info;

int MPIX_Info_init(MPIX_Info** info);
int MPIX_Info_free(MPIX_Info** info);


// If using GPU, specific gpu methods (for either NCCL or HIP)
#ifdef GPU
    __global__ void device_repack(char* __restrict__ sendbuf, 
            char* __restrict__ recvbuf,
            int size_x, int size_y, int size_z);
    void get_mem_types(const void* sendbuf, const void* recvbuf, 
            gpuMemoryType* send_type, gpuMemoryType* recv_type);
    void get_memcpy_kind(gpuMemoryType send_type, gpuMemoryType recv_type, 
            gpuMemcpyKind* memcpy_kind);
    void gpu_repack(int size_i, int size_j, int size_k, char* sendbuf, 
            char* recvbuf);
    void gpu_check(int ierr);
#endif


// General utility methods (that use C++ functions)
void sort(int n_objects, int* object_indices, int* object_values);
void rotate(void* ref, int new_start_byte, int end_byte);
void reverse(void* recvbuf, int n_bytes, int var_bytes);
void repack(int size_i, int size_j, int size_k, char* sendbuf, char* recvbuf);


// Allocate Vector in MPI
int MPIX_Alloc(void** pointer, const int bytes);
int MPIX_Free(void* pointer);

#ifdef __cplusplus
}
#endif


#endif
