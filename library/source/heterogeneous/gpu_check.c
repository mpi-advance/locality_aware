#include "../../../include/utils/gpu_utils.h"
#include "stdio.h"
// Repack Data on Device
#ifdef GPU

void gpu_check(int ierr)
{
    if (ierr != gpuSuccess)
    {
        printf("Error in Device Function!\n");
    }
}
#endif
