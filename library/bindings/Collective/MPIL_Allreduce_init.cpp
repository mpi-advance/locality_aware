#include "collective/allreduce_init.h"
#include "locality_aware.h"
#ifdef GPU
#include "heterogeneous/gpu_collective.h"
#endif

int MPIL_Allreduce_init(const void* sendbuf,
                   void* recvbuf, 
                   int count,
                   MPI_Datatype datatype,
                   MPI_Op op,
                   MPIL_Comm* comm,
                   MPIL_Info* info,
                   MPIL_Request** req_ptr)
{
    if (count == 0)
    {
        return MPI_SUCCESS;
    }

    allreduce_init_ftn method;
#if defined(GPU)
    bool gpu_aware   = false;
    bool copy_to_cpu = false;
#endif

    switch (mpil_allreduce_init_implementation)
    {
#if defined(GPU) 
#if defined(GPU_AWARE)
        case ALLREDUCE_INIT_GPU_RECURSIVE_DOUBLING:
            method = allreduce_init_recursive_doubling;
            gpu_aware = true;
            break;
        case ALLREDUCE_INIT_GPU_DISSEMINATION_LOC:
            method = allreduce_init_dissemination_loc;
            gpu_aware = true;
            break;
        case ALLREDUCE_INIT_GPU_DISSEMINATION_ML:
            method = allreduce_init_dissemination_ml;
            gpu_aware = true;
            break;
        case ALLREDUCE_INIT_GPU_DISSEMINATION_RADIX:
            method = allreduce_init_dissemination_radix;
            gpu_aware = true;
            break;
#if defined(MPI4)
        case ALLREDUCE_INIT_GPU_PMPI:
            method = allreduce_init_pmpi;
            gpu_aware = true;
            break;
#endif
#endif
        case ALLREDUCE_INIT_CTC_RECURSIVE_DOUBLING:
            method = allreduce_init_recursive_doubling;
            copy_to_cpu = true;
            break;
        case ALLREDUCE_INIT_CTC_DISSEMINATION_LOC:
            method = allreduce_init_dissemination_loc;
            copy_to_cpu = true;
            break;
        case ALLREDUCE_INIT_CTC_DISSEMINATION_ML:
            method = allreduce_init_dissemination_ml;
            copy_to_cpu = true;
            break;
        case ALLREDUCE_INIT_CTC_DISSEMINATION_RADIX:
            method = allreduce_init_dissemination_radix;
            copy_to_cpu = true;
            break;
#if defined(MPI4)
        case ALLREDUCE_INIT_CTC_PMPI:
            method = allreduce_init_pmpi;
            copy_to_cpu = true;
            break;
#endif
#endif
        case ALLREDUCE_INIT_RECURSIVE_DOUBLING:
            method = allreduce_init_recursive_doubling;
            break;
        case ALLREDUCE_INIT_DISSEMINATION_LOC:
            method = allreduce_init_dissemination_loc;
            break;
        case ALLREDUCE_INIT_DISSEMINATION_ML:
            method = allreduce_init_dissemination_ml;
            break;
        case ALLREDUCE_INIT_DISSEMINATION_RADIX:
            method = allreduce_init_dissemination_radix;
            break;
#if defined(MPI4)
        case ALLREDUCE_INIT_PMPI:
            method = allreduce_init_pmpi;
            break;
#endif
        default:
#if defined(MPI4)
            method = allreduce_init_pmpi;
#else
            method = allreduce_init_recursive_doubling;
#endif
            break;
    } 

#if defined(GPU)
    if (gpu_aware)
    {
        return gpu_aware_collective(method, sendbuf, recvbuf,
                count, datatype, op, comm, info, req_ptr);
    }
    else if (copy_to_cpu)
    {
        return copy_to_cpu_allreduce_init(method, sendbuf, recvbuf,
                count, datatype, op, comm, info, req_ptr);
    }
#endif

    return method(sendbuf, recvbuf, count, datatype, op, comm, info, req_ptr);
}
