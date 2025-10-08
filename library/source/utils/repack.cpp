#include "../../include/utils/utils.h"
#include <cstring>

// Repack Method (calls device if on GPU)
void repack(int size_i, int size_j, int size_k, char* sendbuf, char* recvbuf)
{
#ifdef GPU
    gpuMemoryType send_type, recv_type;
    get_mem_types(sendbuf, recvbuf, &send_type, &recv_type);

    if (send_type == gpuMemoryTypeDevice && recv_type == gpuMemoryTypeDevice)
    {
        // gpu_repack(size_i, size_j, size_k, sendbuf, recvbuf);
        for (int i = 0; i < size_i; i++)
        {
            for (int j = 0; j < size_j; j++)
            {
                gpuMemcpy(recvbuf + (j * size_i + i) * size_k,
                          sendbuf + (i * size_j + j) * size_k,
                          size_k,
                          gpuMemcpyDeviceToDevice);
            }
        }
    }
    else if (send_type == gpuMemoryTypeDevice)
    {
        for (int i = 0; i < size_i; i++)
        {
            for (int j = 0; j < size_j; j++)
            {
                gpuMemcpy(recvbuf + (j * size_i + i) * size_k,
                          sendbuf + (i * size_j + j) * size_k,
                          size_k,
                          gpuMemcpyHostToDevice);
            }
        }
    }
    else if (recv_type == gpuMemoryTypeDevice)
    {
        for (int i = 0; i < size_i; i++)
        {
            for (int j = 0; j < size_j; j++)
            {
                gpuMemcpy(recvbuf + (j * size_i + i) * size_k,
                          sendbuf + (i * size_j + j) * size_k,
                          size_k,
                          gpuMemcpyDeviceToHost);
            }
        }
    }
    else
#endif
        for (int i = 0; i < size_i; i++)
        {
            for (int j = 0; j < size_j; j++)
            {
                memcpy(recvbuf + (j * size_i + i) * size_k,
                       sendbuf + (i * size_j + j) * size_k,
                       size_k);
            }
        }
}