#include "collective/alltoall.h"
#include "locality_aware.h"
#ifdef GPU
#include "heterogeneous/gpu_collective.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

int MPIL_Alltoall(const void* sendbuf,
                  const int sendcount,
                  MPI_Datatype sendtype,
                  void* recvbuf,
                  const int recvcount,
                  MPI_Datatype recvtype,
                  MPIL_Comm* mpi_comm)
{
    if (sendcount == 0)
    {
        return MPI_SUCCESS;
    }

    alltoall_ftn method;
#if defined(GPU)
    bool gpu_aware   = false;
    bool copy_to_cpu = false;
#endif

    switch (mpil_alltoall_implementation)
    {
#if defined(GPU)
#if defined(GPU_AWARE)
        case ALLTOALL_GPU_PAIRWISE:
            method    = alltoall_pairwise;
            gpu_aware = true;
            break;
        case ALLTOALL_GPU_NONBLOCKING:
            method    = alltoall_nonblocking;
            gpu_aware = true;
            break;
#endif
        case ALLTOALL_CTC_PAIRWISE:
            method      = alltoall_pairwise;
            copy_to_cpu = true;
            break;
        case ALLTOALL_CTC_NONBLOCKING:
            method      = alltoall_nonblocking;
            copy_to_cpu = true;
            break;
#endif
        case ALLTOALL_PAIRWISE:
            method = alltoall_pairwise;
            break;
        case ALLTOALL_NONBLOCKING:
            method = alltoall_nonblocking;
            break;
        case ALLTOALL_HIERARCHICAL_PAIRWISE:
            method = alltoall_hierarchical_pairwise;
            break;
        case ALLTOALL_HIERARCHICAL_NONBLOCKING:
            method = alltoall_hierarchical_nonblocking;
            break;
        case ALLTOALL_MULTILEADER_PAIRWISE:
            method = alltoall_multileader_pairwise;
            break;
        case ALLTOALL_MULTILEADER_NONBLOCKING:
            method = alltoall_multileader_nonblocking;
            break;
        case ALLTOALL_NODE_AWARE_PAIRWISE:
            method = alltoall_node_aware_pairwise;
            break;
        case ALLTOALL_NODE_AWARE_NONBLOCKING:
            method = alltoall_node_aware_nonblocking;
            break;
        case ALLTOALL_LOCALITY_AWARE_PAIRWISE:
            method = alltoall_locality_aware_pairwise;
            break;
        case ALLTOALL_LOCALITY_AWARE_NONBLOCKING:
            method = alltoall_locality_aware_nonblocking;
            break;
        case ALLTOALL_MULTILEADER_LOCALITY_PAIRWISE:
            method = alltoall_multileader_locality_pairwise;
            break;
        case ALLTOALL_MULTILEADER_LOCALITY_NONBLOCKING:
            method = alltoall_multileader_locality_nonblocking;
            break;
        case ALLTOALL_PMPI:
            method = alltoall_pmpi;
            break;
        default:
            method = alltoall_pmpi;
            break;
    }

#if defined(GPU)
    if (gpu_aware)
    {
        return gpu_aware_collective(
            method, sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, mpi_comm);
    }
    else if (copy_to_cpu)
    {
        return copy_to_cpu_alltoall(
            method, sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, mpi_comm);
    }
#endif

    return method(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, mpi_comm);
}

#ifdef __cplusplus
}
#endif
