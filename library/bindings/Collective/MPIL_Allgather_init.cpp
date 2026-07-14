#include "collective/allgather_init.h"
#include "locality_aware.h"
#ifdef GPU
#include "heterogeneous/gpu_collective.h"
#endif

int MPIL_Allgather_init(const void* sendbuf,
                   int sendcount,
                   MPI_Datatype sendtype,
                   void* recvbuf, 
                   int recvcount,
                   MPI_Datatype recvtype,
                   MPIL_Comm* comm,
                   MPIL_Info* info,
                   MPIL_Request** req_ptr)
{
    if (sendcount == 0) 
    {
        return MPI_SUCCESS;
    }

    allgather_init_ftn method;
#if defined(GPU)
    bool gpu_aware   = false;
    bool copy_to_cpu = false;
#endif

    switch (mpil_allgather_init_implementation)
    {
#if defined(GPU) 
#if defined(GPU_AWARE)
        case ALLGATHER_INIT_GPU_RING:
            method = allgather_init_ring;
            gpu_aware = true;
            break;
        case ALLGATHER_INIT_GPU_BRUCK:
            method = allgather_init_bruck;
            gpu_aware = true;
            break;
#if defined(MPI4)
        case ALLGATHER_INIT_GPU_PMPI:
            method = allgather_init_pmpi;
            gpu_aware = true;
            break;
#endif
#endif
        case ALLGATHER_INIT_CTC_RING:
            method = allgather_init_ring;
            copy_to_cpu = true;
            break;
        case ALLGATHER_INIT_CTC_BRUCK:
            method = allgather_init_bruck;
            copy_to_cpu = true;
            break;
#if defined(MPI4)
        case ALLGATHER_INIT_CTC_PMPI:
            method = allgather_init_pmpi;
            copy_to_cpu = true;
            break;
#endif
#endif
        case ALLGATHER_INIT_RING:
            method = allgather_init_ring;
            break;
        case ALLGATHER_INIT_BRUCK:
            method = allgather_init_bruck;
            break;
#if defined(MPI4)
        case ALLGATHER_INIT_PMPI:
            method = allgather_init_pmpi;
            break;
#endif
        default:
#if defined(MPI4)
            method = allgather_init_pmpi;
#else
            if (sendcount > 1024)
            {
                method = allgather_init_ring;
            }
            else
            {
                method = allgather_init_bruck;
            }
#endif
            break;
    } 

#if defined(GPU)
    if (gpu_aware)
    {
        return gpu_aware_collective(method, sendbuf, sendcount, sendtype,
                recvbuf, recvcount, recvtype, comm, info, req_ptr);
    }
    else if (copy_to_cpu)
    {
        return copy_to_cpu_allgather_init(method, sendbuf, sendcount, sendtype,
                recvbuf, recvcount, recvtype, comm, info, req_ptr);
    }           
#endif

    return method(sendbuf, sendcount, sendtype, recvbuf, recvcount, 
            recvtype, comm, info, req_ptr);
}
