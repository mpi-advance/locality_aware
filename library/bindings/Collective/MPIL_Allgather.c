#include "collective/allgather.h"
#include "locality_aware.h"
#ifdef GPU
#include "heterogeneous/gpu_allgather.h"
#endif

int MPIL_Allgather(const void* sendbuf,
                   int sendcount,
                   MPI_Datatype sendtype,
                   void* recvbuf, 
                   int recvcount,
                   MPI_Datatype recvtype,
                   MPIL_Comm* comm)
{
    allgather_ftn method;

    switch (mpil_allgather_implementation)
    {
#if defined(GPU) 
#if defined(GPU_AWARE)
        case ALLGATHER_GPU_RING:
            method = gpu_aware_allgather_ring;
            break;
        case ALLGATHER_GPU_BRUCK:
            method = gpu_aware_allgather_bruck;
            break;
        case ALLGATHER_GPU_PMPI:
            method = gpu_aware_allgather_pmpi;
            break;
#endif
        case ALLGATHER_CTC_RING:
            method = copy_to_cpu_allgather_ring;
            break;
        case ALLGATHER_CTC_BRUCK:
            method = copy_to_cpu_allgather_bruck;
            break;
        case ALLGATHER_CTC_PMPI:
            method = copy_to_cpu_allgather_pmpi;
            break;
#endif
        case ALLGATHER_RING:
            method = allgather_ring;
            break;
        case ALLGATHER_BRUCK:
            method = allgather_bruck;
            break;
        case ALLGATHER_PMPI:
            method = allgather_pmpi;
            break;
        default:
            method = allgather_pmpi;
            break;
    } 

    return method(sendbuf, sendcount, sendtype, recvbuf, recvcount, 
            recvtype, comm);
}

