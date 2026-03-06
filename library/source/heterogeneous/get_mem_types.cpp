#include "heterogeneous/gpu_utils.h"
// GPU Method to find where memory was allocated
#ifdef GPU
void get_mem_types(const void* sendbuf,
                   const void* recvbuf,
                   gpuMemoryType* send_ptr,
                   gpuMemoryType* recv_ptr)
{
    gpuMemoryType send_type, recv_type;

    gpuPointerAttributes mem;
    gpuPointerGetAttributes(&mem, sendbuf);
    int ierr = gpuGetLastError();
    if (ierr == gpuErrorInvalidValue)
    {
        send_type = gpuMemoryTypeHost;
    }
    else
    {
        send_type = mem.type;
    }

    gpuPointerGetAttributes(&mem, recvbuf);
    ierr = gpuGetLastError();
    if (ierr == gpuErrorInvalidValue)
    {
        recv_type = gpuMemoryTypeHost;
    }
    else
    {
        recv_type = mem.type;
    }

    *send_ptr = send_type;
    *recv_ptr = recv_type;
}
#endif