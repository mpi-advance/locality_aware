#include "collective/allreduce.h"
#include "locality_aware.h"
#ifdef GPU
#include "heterogeneous/gpu_allreduce.h"
#endif

int MPIL_Allreduce(const void* sendbuf,
                   void* recvbuf, 
                   int count,
                   MPI_Datatype datatype,
                   MPI_Op op,
                   MPIL_Comm* comm)
{
    allreduce_ftn method;

    switch (mpil_allreduce_implementation)
    {
#if defined(GPU) 
#if defined(GPU_AWARE)
        case ALLREDUCE_GPU_RECURSIVE_DOUBLING:
            method = gpu_aware_allreduce_recursive_doubling;
            break;
        case ALLREDUCE_GPU_DISSEMINATION_LOC:
            method = gpu_aware_allreduce_dissemination_loc;
            break;
        case ALLREDUCE_GPU_DISSEMINATION_ML:
            method = gpu_aware_allreduce_dissemination_ml;
            break;
        case ALLREDUCE_GPU_DISSEMINATION_RADIX:
            method = gpu_aware_allreduce_dissemination_radix;
            break;
        case ALLREDUCE_GPU_PMPI:
            method = gpu_aware_allreduce_pmpi;
            break;
#endif
        case ALLREDUCE_CTC_RECURSIVE_DOUBLING:
            method = copy_to_cpu_allreduce_recursive_doubling;
            break;
        case ALLREDUCE_CTC_DISSEMINATION_LOC:
            method = copy_to_cpu_allreduce_dissemination_loc;
            break;
        case ALLREDUCE_CTC_DISSEMINATION_ML:
            method = copy_to_cpu_allreduce_dissemination_ml;
            break;
        case ALLREDUCE_CTC_DISSEMINATION_RADIX:
            method = copy_to_cpu_allreduce_dissemination_radix;
            break;
        case ALLREDUCE_CTC_PMPI:
            method = copy_to_cpu_allreduce_pmpi;
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

    return method(sendbuf, recvbuf, count, datatype, op, comm);
}

