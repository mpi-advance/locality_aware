#include "../../../include/utils/gpu_utils.h"

// Repack Data on Device
#ifdef GPU
__global__ void device_repack(char* __restrict__ sendbuf,
                              char* __restrict__ recvbuf,
                              int size_x,
                              int size_y,
                              int size_z)
{
    const int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    const int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    const int tid_z = threadIdx.z + blockIdx.z * blockDim.z;

    if (tid_x >= size_x || tid_y >= size_y || tid_z >= size_z)
    {
        return;
    }

    recvbuf[(tid_y * size_x + tid_x) * size_z + tid_z] =
        sendbuf[(tid_x * size_y + tid_y) * size_z + tid_z];
}
#endif