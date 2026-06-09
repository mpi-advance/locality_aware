#include "collective/allreduce.h"
#include "locality_aware.h"
#ifdef GPU
#include "heterogeneous/gpu_collective.h"
#endif

int MPIL_Allreduce(const void* sendbuf,
                   void* recvbuf,
                   int count,
                   MPI_Datatype datatype,
                   MPI_Op op,
                   MPIL_Comm* comm)
{
    if (count == 0)
    {
        return MPI_SUCCESS;
    }

    allreduce_ftn method;
#if defined(GPU)
    bool gpu_aware   = false;
    bool copy_to_cpu = false;
#endif

    switch (mpil_allreduce_implementation)
    {
#if defined(GPU)
#if defined(GPU_AWARE)
        case ALLREDUCE_GPU_RECURSIVE_DOUBLING:
            method    = allreduce_recursive_doubling;
            gpu_aware = true;
            break;
        case ALLREDUCE_GPU_DISSEMINATION_LOC:
            method    = allreduce_dissemination_loc;
            gpu_aware = true;
            break;
        case ALLREDUCE_GPU_DISSEMINATION_ML:
            method    = allreduce_dissemination_ml;
            gpu_aware = true;
            break;
        case ALLREDUCE_GPU_DISSEMINATION_RADIX:
            method    = allreduce_dissemination_radix;
            gpu_aware = true;
            break;
        case ALLREDUCE_GPU_PMPI:
            method    = allreduce_pmpi;
            gpu_aware = true;
            break;
#endif
        case ALLREDUCE_CTC_RECURSIVE_DOUBLING:
            method      = allreduce_recursive_doubling;
            copy_to_cpu = true;
            break;
        case ALLREDUCE_CTC_DISSEMINATION_LOC:
            method      = allreduce_dissemination_loc;
            copy_to_cpu = true;
            break;
        case ALLREDUCE_CTC_DISSEMINATION_ML:
            method      = allreduce_dissemination_ml;
            copy_to_cpu = true;
            break;
        case ALLREDUCE_CTC_DISSEMINATION_RADIX:
            method      = allreduce_dissemination_radix;
            copy_to_cpu = true;
            break;
        case ALLREDUCE_CTC_PMPI:
            method      = allreduce_pmpi;
            copy_to_cpu = true;
            break;
#endif
        case ALLREDUCE_RECURSIVE_DOUBLING:
            method = allreduce_recursive_doubling;
            break;
        case ALLREDUCE_DISSEMINATION_LOC:
            method = allreduce_dissemination_loc;
            break;
        case ALLREDUCE_DISSEMINATION_ML:
            method = allreduce_dissemination_ml;
            break;
        case ALLREDUCE_DISSEMINATION_RADIX:
            method = allreduce_dissemination_radix;
            break;
        case ALLREDUCE_PMPI:
            method = allreduce_pmpi;
            break;
        default:
            method = allreduce_pmpi;
            break;
    }

#if defined(GPU)
    if (gpu_aware)
    {
        return gpu_aware_collective(method, sendbuf, recvbuf, count, datatype, op, comm);
    }
    else if (copy_to_cpu)
    {
        return copy_to_cpu_allreduce(method, sendbuf, recvbuf, count, datatype, op, comm);
    }
#endif

    return method(sendbuf, recvbuf, count, datatype, op, comm);
}
