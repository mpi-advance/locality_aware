#ifndef MPI_ADVANCE_UTILS_H
#define MPI_ADVANCE_UTILS_H

#ifdef HIP
#include "utils_hip.h"
#endif

#ifdef CUDA
#include "utils_cuda.h"
#endif

////---why was this in the .cpp file?
#ifdef HIP
#include "hip/hip_runtime.h"
#endif
////


#ifdef __cplusplus
extern "C" {
#endif

// MPIL Info Object
/** @brief MPIL_Info object 
	\todo why this instead of just an entry in a normal MPI_Info object?
**/
typedef struct _MPIL_Info
{
    int crs_num_initialized;
    int crs_size_initialized;
} MPIL_Info;

/** @brief Constructor of MPIL_Info object, initialized values = 0**/
int MPIL_Info_init(MPIL_Info** info);

/** @brief deallocate and delete supplied info object **/
int MPIL_Info_free(MPIL_Info** info);

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

// General utility methods (that use C++ functions)

/** @brief wrapper around std::sort 
	@param [in] n_objects number of objects to short
	@param [in, out] array of indexes
	@param [in] array of values 
**/
void sort(int n_objects, int* object_indices, int* object_values);

/** @brief wrapper around std::rotate,
 *	@details
 *      Rotates such that new_first_byte is first in array
 *		Divides recvbuf into two parts [first, middle] and (middle, last)
 *		then swaps their positioning. 
 *		Example: A = 0, 1, 2, 3, 4, 5
 *           std::rotate(A*, 2, A*+6) would split into (0, 1) and (2, 3, 4, 5)
 *			 and after running A = 2, 3, 4, 5, 0, 1
 *	
 *	@param [in, out] recvbuf buffer of elements to rotate
 *	@param [in] new_first_byte index immediately after the split point. 
 *	@param [in] index of last element in the effected range 
**/
void rotate(void* ref, int new_start_byte, int end_byte);

/** @brief reverses order of elements in recv_buffer
 *	@details
 *		Divides recvbuf into two parts [first, middle] and (middle, last)
 *		then swaps their positioning. 
 *		Example: A = 0, 1, 2, 3, 4, 5
 *           std::rotate(A*, 2, A*+6) would split into (0, 1) and (2, 3, 4, 5)
 *			 and after running A = 2, 3, 4, 5, 0, 1
 *	
 *	@param [in, out] recvbuf buffer of elements to rotate
 *	@param [in] new_first_byte index immediately after the split point. 
 *	@param [in] index of last element in the effected range 
 *  \todo why this instead of std::reverse?
**/
void reverse(void* recvbuf, int n_bytes, int var_bytes);


void repack(int size_i, int size_j, int size_k, char* sendbuf, char* recvbuf);

// Allocate Vector in MPI
int MPIL_Alloc(void** pointer, const int bytes);
int MPIL_Free(void* pointer);

#ifdef __cplusplus
}
#endif

#endif
