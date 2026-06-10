#include "collective/allgather.h"
#include "locality_aware.h"
#ifdef GPU
#include "heterogeneous/gpu_collective.h"
#endif

int MPIL_Allgather(const void* sendbuf,
                   int sendcount,
                   MPI_Datatype sendtype,
                   void* recvbuf, 
                   int recvcount,
                   MPI_Datatype recvtype,
                   MPIL_Comm* comm)
{
    if (sendcount == 0) 
    {
        return MPI_SUCCESS;
    }

    allgather_ftn method;
    bool gpu_aware = false;
    bool copy_to_cpu = false;

    switch (mpil_allgather_implementation)
    {
#if defined(GPU) 
#if defined(GPU_AWARE)
        case ALLGATHER_GPU_RING:
            method = allgather_ring;
            gpu_aware = true;
            break;
        case ALLGATHER_GPU_BRUCK:
            method = allgather_bruck;
            gpu_aware = true;
            break;
        case ALLGATHER_GPU_PMPI:
            method = allgather_pmpi;
            gpu_aware = true;
            break;
#endif
        case ALLGATHER_CTC_RING:
            method = allgather_ring;
            copy_to_cpu = true;
            break;
        case ALLGATHER_CTC_BRUCK:
            method = allgather_bruck;
            copy_to_cpu = true;
            break;
        case ALLGATHER_CTC_PMPI:
            method = allgather_pmpi;
            copy_to_cpu = true;
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

#if defined(GPU)
    if (gpu_aware)
    {
        return gpu_aware_collective(method, sendbuf, sendcount, sendtype,
                recvbuf, recvcount, recvtype, comm);
    }
    else if (copy_to_cpu)
    {
        return copy_to_cpu_allgather(method, sendbuf, sendcount, sendtype,
                recvbuf, recvcount, recvtype, comm);
    }           
#endif

    return method(sendbuf, sendcount, sendtype, recvbuf, recvcount, 
            recvtype, comm);
}
