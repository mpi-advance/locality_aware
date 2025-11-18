#include "locality_aware.h"

#if defined(GPU)
#include "heterogeneous/gpu_utils.h"
#endif

int MPIL_Free(void* pointer)
{
    if (pointer != nullptr)
    {
        char* char_ptr = (char*)pointer;
        delete[] char_ptr;
    }

    return MPI_SUCCESS;
}

#if defined(GPU)
int MPIL_GPU_Free(void* pointer)
{
    if (pointer != nullptr)
    {
        gpuFree(pointer);
    }

    return MPI_SUCCESS;
}
#endif
