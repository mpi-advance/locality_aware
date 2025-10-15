#include "locality_aware.h"
#include "collective/alltoall.h"

// Default alltoall is pairwise
//AlltoallMethod mpil_alltoall_implementation = ALLTOALL_PAIRWISE;

/**************************************************
 * Locality-Aware Point-to-Point Alltoall
 *  - Aggregates messages locally to reduce
 *      non-local communication
 *  - First redistributes on-node so that each
 *      process holds all data for a subset
 *      of other nodes
 *  - Then, performs inter-node communication
 *      during which each process exchanges
 *      data with their assigned subset of nodes
 *  - Finally, redistribute received data
 *      on-node so that each process holds
 *      the correct final data
 *************************************************/
 
 int MPIL_Alltoall(const void* sendbuf,
                  const int sendcount,
                  MPI_Datatype sendtype,
                  void* recvbuf,
                  const int recvcount,
                  MPI_Datatype recvtype,
                  MPIL_Comm* mpi_comm)
{
#ifdef GPU
#ifdef GPU_AWARE
    return gpu_aware_alltoall(alltoall_pairwise,
                              sendbuf,
                              sendcount,
                              sendtype,
                              recvbuf,
                              recvcount,
                              recvtype,
                              mpi_comm);
#endif
#endif
    alltoall_ftn method;

    switch (mpil_alltoall_implementation)
    {
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
