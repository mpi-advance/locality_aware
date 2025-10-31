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
/** @brief copy from send_buf to recv_buff from all threads. **/
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
					 
/** @brief set dims then launch device_repack
\todo why all dims 8?**/
void gpu_repack(int size_i, int size_j, int size_k, char* sendbuf, char* recvbuf);

/** @brief prints error message if ierr is not gpuSuccess **/
void gpu_check(int ierr);

/** @brief copy contents from from recvbuf to send_buf
 * @details
 *   uses size_i, size_j and size_k only if using GPU
 *   copies data from sendbuf into recvbuf
 *
 * @param [in] size_i if GPU
 * @param [in] size_j if GPU
 * @param [in] size_k 
 * @param [out]sendbuf
 * @param [in] recvbuf 
**/
void repack(int size_i, int size_j, int size_k, char* sendbuf, char* recvbuf);

#endif

#ifdef __cplusplus
}
#endif

#endif
