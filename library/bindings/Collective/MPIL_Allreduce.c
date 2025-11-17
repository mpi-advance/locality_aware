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
        case ALLREDUCE_GPU_DISSEMINATION:
            method = gpu_aware_allreduce_dissemination;
            break;
        case ALLREDUCE_GPU_DISSEMINTATION_LOC:
            method = gpu_aware_allreduce_dissemination_loc;
            break;
#endif
        case ALLREDUCE_CTC_RECURSIVE_DOUBLING:
            method = copy_to_cpu_allreduce_recursive_doubling;
            break;
        case ALLREDUCE_CTC_DISSEMINATION:
            method = copy_to_cpu_allreduce_dissemination;
            break;
        case ALLREDUCE_CTC_DISSEMINTATION_LOC:
            method = copy_to_cpu_allreduce_dissemination_loc;
            break;
#endif

        case ALLREDUCE_RECURSIVE_DOUBLING:
            method = allreduce_recursive_doubling;
            break;
        case ALLREDUCE_DISSEMINATION:
            method = allreduce_dissemination;
            break;
        case ALLREDUCE_DISSEMINATION_LOC:
            method = allreduce_dissemination_loc;
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

