#include "utils.h"
#include <algorithm>

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

#ifdef GPU
void get_mem_types(const void* sendbuf, const void* recvbuf, cudaMemoryType& send_type,
        cudaMemoryType& recv_type)
{
    cudaPointerAttributes mem;
    cudaPointerGetAttributes(&mem, sendbuf);
    int ierr = cudaGetLastError();
    if (ierr == cudaErrorInvalidValue)
        send_type = cudaMemoryTypeHost;
    else
        send_type = mem.type;

    cudaPointerGetAttributes(&mem, recvbuf);
    ierr = cudaGetLastError();
    if (ierr == cudaErrorInvalidValue)
        recv_type = cudaMemoryTypeHost;
    else
        recv_type = mem.type;
}
#endif
