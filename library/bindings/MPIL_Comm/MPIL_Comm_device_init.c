#include "communicator/MPIL_Comm.h"

int MPIL_Comm_device_init(MPIL_Comm* xcomm)
{
#ifdef GPU
    if (xcomm->local_comm == MPI_COMM_NULL)
    {
        MPIL_Comm_topo_init(xcomm);
    }

    int local_rank, ierr;
    MPI_Comm_rank(xcomm->local_comm, &local_rank);
    ierr = gpuGetDeviceCount(&(xcomm->gpus_per_node));
    gpu_check(ierr);
    if (xcomm->gpus_per_node)
    {
        xcomm->rank_gpu = local_rank;
        ierr            = gpuStreamCreate(&(xcomm->proc_stream));
        gpu_check(ierr);
    }
#endif

    return MPI_SUCCESS;
}
