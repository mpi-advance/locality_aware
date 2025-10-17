#include "locality_aware.h"

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
