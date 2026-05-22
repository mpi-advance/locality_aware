#include "communicator/MPIL_Comm.hpp"
#include "locality_aware.h"
#ifdef GPU
#include "heterogeneous/gpu_utils.h"
#endif

int MPIL_Comm_device_init(MPIL_Comm* xcomm)
{
#ifdef GPU
    if (xcomm->local_comm == MPI_COMM_NULL)
    {
        MPIL_Comm_topo_init(xcomm);
    }

    int local_rank, ppn, ierr;
    MPI_Comm_rank(xcomm->local_comm, &local_rank);
    MPI_Comm_size(xcomm->local_comm, &ppn);
    ierr = gpuGetDeviceCount(&(xcomm->gpus_per_node));
    gpu_check(ierr);
    if (xcomm->gpus_per_node == ppn)
    {
        xcomm->rank_gpu = local_rank;
        ierr            = gpuSetDevice(local_rank);
        gpu_check(ierr);
        ierr            = gpuStreamCreate((gpuStream_t*)&(xcomm->proc_stream));
        gpu_check(ierr);
    }
    else if (xcomm->gpus_per_node == 1)
    {
        xcomm->rank_gpu = 0;
        ierr            = gpuSetDevice(0);
        gpu_check(ierr);
        ierr            = gpuStreamCreate((gpuStream_t*)&(xcomm->proc_stream));
        gpu_check(ierr);
    }
        
#endif

    return MPI_SUCCESS;
}
