#include "locality_aware.h"

#ifdef __cplusplus
extern "C" {
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
    gpuFree(pointer);

    return MPI_SUCCESS;
}
#endif

#ifdef __cplusplus
}
#endif
