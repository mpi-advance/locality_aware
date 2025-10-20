#include "heterogeneous/gpu_utils.h"
#include "stdio.h"

// Repack Data on Device
void gpu_check(int ierr)
{
    if (ierr != gpuSuccess)
    {
        printf("Error in Device Function!\n");
    }
}

