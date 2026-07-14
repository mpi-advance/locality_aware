#include "locality_aware.h"

#ifdef __cplusplus
extern "C" {
#endif

int MPIL_Alloc(void** pointer, const int bytes)
{
    if (bytes == 0)
    {
        *pointer = nullptr;
    }
    else
    {
        *pointer = new char[bytes];
    }

    return MPI_SUCCESS;
}

#if defined(GPU)
int MPIL_GPU_Alloc(void** pointer, const int bytes)
{
    if (bytes == 0)
    {
        *pointer = nullptr;
    }
    else
    {
        gpuMalloc((void**)pointer, bytes);
    }

    return MPI_SUCCESS;
}
#endif

#ifdef __cplusplus
}
#endif
