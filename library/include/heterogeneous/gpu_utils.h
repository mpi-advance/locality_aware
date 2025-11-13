#ifndef MPIL_GPU_UTILS_H
#define MPIL_GPU_UTILS_H

#ifdef HIP
#include "utils_hip.h"
#endif

#ifdef CUDA
#include "utils_cuda.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

// If using GPU, specific gpu methods (for either NCCL or HIP)
#ifdef GPU
/** @brief Kernel for copying from send_buf into recv_buf using all GPU threads. **/
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

/** @brief Set dims (currently 8,8,8) then launch ::device_repack() kernel **/
void gpu_repack(int size_i, int size_j, int size_k, char* sendbuf, char* recvbuf);

/** @brief Prints error message if ierr is not gpuSuccess **/
void gpu_check(int ierr);

/** @brief Copy contents from sendbuf into recvbuf
 * @details Checks the type of memory involved in order to make sure the correct method
 * for copying the data is called. `memcpy` is used for CPU transfers; `::gpuMemcpy` is
 * used for GPU transfers.
 * @param [in] size_i if GPU
 * @param [in] size_j if GPU
 * @param [in] size_k
 * @param [out] sendbuf
 * @param [in] recvbuf
 **/
void repack(int size_i, int size_j, int size_k, char* sendbuf, char* recvbuf);

#endif

#ifdef __cplusplus
}
#endif

#endif
