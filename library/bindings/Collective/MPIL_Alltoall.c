#include "collective/alltoall.h"
#include "locality_aware.h"
#ifdef GPU
#include "heterogeneous/gpu_alltoall.h"
#endif

int MPIL_Alltoall(const void* sendbuf,
                  const int sendcount,
                  MPI_Datatype sendtype,
                  void* recvbuf,
                  const int recvcount,
                  MPI_Datatype recvtype,
                  MPIL_Comm* mpi_comm)
{
    alltoall_ftn method;

    switch (mpil_alltoall_implementation)
    {
#ifdef GPU
#ifdef GPU_AWARE

        case ALLTOALL_GPU_PAIRWISE:
            method = gpu_aware_alltoall_pairwise;
            break;
        case ALLTOALL_GPU_NONBLOCKING:
            method = gpu_aware_alltoall_nonblocking;
            break;
        case ALLTOALL_CTC_PAIRWISE:
            method = copy_to_cpu_alltoall_pairwise;
            break;
        case ALLTOALL_CTC_NONBLOCKING:
            method = copy_to_cpu_alltoall_nonblocking;
            break;
#endif
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

    return method(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, mpi_comm);
}
