#include "utils.h"
#include <algorithm>
#include <cstring>
#include "mpi.h"
#include "stdio.h"

#ifdef HIP
#include "hip/hip_runtime.h"
#endif

// MPIX Info Object Routines
int MPIX_Info_init(MPIX_Info** info_ptr)
{
    MPIX_Info* xinfo = (MPIX_Info*)malloc(sizeof(MPIX_Info));
    xinfo->crs_num_initialized = 0;
    xinfo->crs_size_initialized = 0;

    *info_ptr = xinfo;

    return MPI_SUCCESS;
}

int MPIX_Info_free(MPIX_Info** info_ptr)
{
    MPIX_Info* xinfo = *info_ptr;
    free(xinfo);

    return MPI_SUCCESS;
}


void sort(int n_objects, int* object_indices, int* object_values)
{
    std::sort(object_indices, object_indices+n_objects,
            [&](const int i, const int j)
            {
                return object_values[i] > object_values[j];
            });
}

void rotate(void* recvbuf,
        int new_first_byte,
        int last_byte)
{
    char* recv_buffer = (char*)(recvbuf);
    std::rotate(recv_buffer, &(recv_buffer[new_first_byte]), &(recv_buffer[last_byte]));
} 

void reverse(void* recvbuf, int n_bytes, int var_bytes)
{
    char* recv_buffer = (char*)(recvbuf);
    int n_vars = n_bytes / var_bytes;
    for (int i = 0; i < n_vars / 2; i++)
        for (int j = 0; j < var_bytes; j++)
            std::swap(recv_buffer[i*var_bytes+j], recv_buffer[(n_vars-i-1)*var_bytes+j]);
}



// Repack Data on Device
#ifdef GPU
__global__ void device_repack(char* __restrict__ sendbuf, char* __restrict__ recvbuf,
        int size_x, int size_y, int size_z)
{
    const int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    const int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    const int tid_z = threadIdx.z + blockIdx.z * blockDim.z;

    if (tid_x >= size_x || tid_y >= size_y || tid_z >= size_z)
        return;

    recvbuf[(tid_y*size_x+tid_x)*size_z+tid_z] =
            sendbuf[(tid_x*size_y+tid_y)*size_z+tid_z];
}

void gpu_repack(int size_i, int size_j, int size_k, char* sendbuf, char* recvbuf)
{
    dim3 dimBlock(8, 8, 8);
    int grid_x = ((size_i - 1) / 8) + 1;
    int grid_y = ((size_j - 1) / 8) + 1;
    int grid_z = ((size_k - 1) / 8) + 1;
    dim3 dimGrid(grid_x, grid_y, grid_z);
    device_repack<<<dimGrid, dimBlock>>>(sendbuf, recvbuf, size_i, size_j, size_k);
}

void gpu_check(int ierr)
{
    if (ierr != gpuSuccess)
        printf("Error in Device Function!\n");
}
#endif

// Repack Method (calls device if on GPU)
void repack(int size_i, int size_j, int size_k, char* sendbuf, char* recvbuf)
{
#ifdef GPU
    gpuMemoryType send_type, recv_type;
    get_mem_types(sendbuf, recvbuf, &send_type, &recv_type);

    if (send_type == gpuMemoryTypeDevice &&
            recv_type == gpuMemoryTypeDevice)
	{
		//gpu_repack(size_i, size_j, size_k, sendbuf, recvbuf);
		for (int i = 0; i < size_i; i++)
			for (int j = 0; j < size_j; j++)
				gpuMemcpy(recvbuf + (j*size_i+i)*size_k,
					sendbuf + (i*size_j+j)*size_k,
					size_k, gpuMemcpyDeviceToDevice);
	}
    else if (send_type == gpuMemoryTypeDevice)
	{
    	for (int i = 0; i < size_i; i++)
    		for (int j = 0; j < size_j; j++)
 				gpuMemcpy(recvbuf + (j*size_i+i)*size_k,
					sendbuf + (i*size_j+j)*size_k,
  					size_k, gpuMemcpyHostToDevice);
	}
	else if (recv_type == gpuMemoryTypeDevice)
	{
    	for (int i = 0; i < size_i; i++)
    		for (int j = 0; j < size_j; j++)
 				gpuMemcpy(recvbuf + (j*size_i+i)*size_k,
					sendbuf + (i*size_j+j)*size_k,
  					size_k, gpuMemcpyDeviceToHost);
	}
	else
#endif	
    for (int i = 0; i < size_i; i++)
    	for (int j = 0; j < size_j; j++)
 			memcpy(recvbuf + (j*size_i+i)*size_k,
					sendbuf + (i*size_j+j)*size_k,
  					size_k);
}


// GPU Method to find where memory was allocated
#ifdef GPU
void get_mem_types(const void* sendbuf, const void* recvbuf, 
        gpuMemoryType* send_ptr, gpuMemoryType* recv_ptr)
{
    gpuMemoryType send_type, recv_type;

    gpuPointerAttributes mem;
    gpuPointerGetAttributes(&mem, sendbuf);
    int ierr = gpuGetLastError();
    if (ierr == gpuErrorInvalidValue)
        send_type = gpuMemoryTypeHost;
    else
        send_type = mem.type;

    gpuPointerGetAttributes(&mem, recvbuf);
    ierr = gpuGetLastError();
    if (ierr == gpuErrorInvalidValue)
        recv_type = gpuMemoryTypeHost;
    else
        recv_type = mem.type;

    *send_ptr = send_type;
    *recv_ptr = recv_type;
}

void get_memcpy_kind(gpuMemoryType send_type, gpuMemoryType recv_type, 
        gpuMemcpyKind* memcpy_kind)
{
    if (send_type == gpuMemoryTypeDevice &&
            recv_type == gpuMemoryTypeDevice)
        *memcpy_kind = gpuMemcpyDeviceToDevice;
    else if (send_type == gpuMemoryTypeDevice)
        *memcpy_kind = gpuMemcpyDeviceToHost;
    else if (recv_type == gpuMemoryTypeDevice)
        *memcpy_kind = gpuMemcpyHostToDevice;
    else
        *memcpy_kind = gpuMemcpyHostToHost;
}
#endif




int MPIX_Alloc(void** pointer, const int bytes)
{
    if (bytes == 0)
        *pointer = NULL;
    else
        *pointer = new char[bytes];

    return MPI_SUCCESS;
}
int MPIX_Free(void* pointer)
{
    if (pointer != NULL)
	{
		char* char_ptr = (char*)pointer;
        delete[] char_ptr;
	}
    
    return MPI_SUCCESS;
}
