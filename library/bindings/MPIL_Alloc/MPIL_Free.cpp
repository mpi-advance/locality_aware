#include "locality_aware.h"

int MPIL_Free(void* pointer)
{
    if (pointer != nullptr)
    {
        char* char_ptr = (char*)pointer;
        delete[] char_ptr;
    }

    return MPI_SUCCESS;
}
