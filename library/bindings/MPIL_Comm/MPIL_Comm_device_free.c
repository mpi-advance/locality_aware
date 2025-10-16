#include "communicator/MPIL_Comm.h"
#include "locality_aware.h"

#ifdef GPU
#include "heterogeneous/gpu_utils.h"
#endif

int MPIL_Comm_device_free(MPIL_Comm* xcomm)
{
#ifdef GPU
    int ierr = gpuSuccess;
    if (xcomm->gpus_per_node)
    {
        ierr = gpuStreamDestroy(xcomm->proc_stream);
    }
    gpu_check(ierr);
#endif

    return MPI_SUCCESS;
}
