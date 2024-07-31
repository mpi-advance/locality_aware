#include "utils.h"
#include <algorithm>
#include <cstring>
#include "mpi.h"


// MPIX Info Object Routines
int MPIX_Info_init(MPIX_Info** info_ptr)
{
    MPIX_Info* xinfo = (MPIX_Info*)malloc(sizeof(MPIX_Info));
    int flag;
    MPI_Comm_get_attr( MPI_COMM_WORLD, MPI_TAG_UB, &(xinfo->max_tag), &flag);
    xinfo->tag = 159 % xinfo->max_tag;
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

int MPIX_Info_tag(MPIX_Info* xinfo, int* tag)
{
    *tag = xinfo->tag;
    xinfo->tag = ((xinfo->tag + 1 )% xinfo->max_tag);

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
#endif
