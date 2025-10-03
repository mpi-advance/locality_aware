#ifndef MPIL_GPU_UTILS_H
#define MPIL_GPU_UTILS_H

/* #ifdef HIP
#include "utils_hip.h"

#endif */

/* #ifdef CUDA

#include "utils_cuda.h"

#endif */

#ifdef __cplusplus
extern "C" {
#endif


// If using GPU, specific gpu methods (for either NCCL or HIP)
#ifdef GPU
__global__ void device_repack(char* __restrict__ sendbuf,
                              char* __restrict__ recvbuf,
                              int size_x,
                              int size_y,
                              int size_z);
void get_mem_types(const void* sendbuf,
                   const void* recvbuf,
                   gpuMemoryType* send_type,
                   gpuMemoryType* recv_type);
void get_memcpy_kind(gpuMemoryType send_type,
                     gpuMemoryType recv_type,
                     gpuMemcpyKind* memcpy_kind);
void gpu_repack(int size_i, int size_j, int size_k, char* sendbuf, char* recvbuf);
void gpu_check(int ierr);
#endif



#ifdef __cplusplus
}
#endif

#endif
