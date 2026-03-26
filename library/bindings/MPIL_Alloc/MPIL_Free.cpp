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

#ifdef __cplusplus
}
#endif
