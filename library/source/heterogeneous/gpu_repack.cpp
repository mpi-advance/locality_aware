#include "heterogeneous/gpu_utils.h"

#ifdef GPU
void gpu_repack(int size_i, int size_j, int size_k, char* sendbuf, char* recvbuf)
{
    dim3 dimBlock(8, 8, 8);
    int grid_x = ((size_i - 1) / 8) + 1;
    int grid_y = ((size_j - 1) / 8) + 1;
    int grid_z = ((size_k - 1) / 8) + 1;
    dim3 dimGrid(grid_x, grid_y, grid_z);
    device_repack<<<dimGrid, dimBlock>>>(sendbuf, recvbuf, size_i, size_j, size_k);
}
#endif