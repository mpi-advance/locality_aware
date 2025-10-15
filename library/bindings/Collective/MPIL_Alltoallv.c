#include "locality_aware.h"
#include "collective/alltoallv.h"

// Default alltoallv is pairwise
//enum AlltoallvMethod mpil_alltoallv_implementation = ALLTOALLV_PAIRWISE;

/**************************************************
 * Locality-Aware Point-to-Point Alltoallv
 * Same as PMPI_Alltoall (no load balancing)
 *  - Aggregates messages locally to reduce
 *      non-local communciation
 *  - First redistributes on-node so that each
 *      process holds all data for a subset
 *      of other nodes
 *  - Then, performs inter-node communication
 *      during which each process exchanges
 *      data with their assigned subset of nodes
 *  - Finally, redistribute received data
 *      on-node so that each process holds
 *      the correct final data
 *  - To be used when sizes are relatively balanced
 *  - For load balacing, use persistent version
 *      - Load balacing is too expensive for
 *          non-persistent Alltoallv
 *************************************************/
int MPIL_Alltoallv(const void* sendbuf,
                   const int sendcounts[],
                   const int sdispls[],
                   MPI_Datatype sendtype,
                   void* recvbuf,
                   const int recvcounts[],
                   const int rdispls[],
                   MPI_Datatype recvtype,
                   MPIL_Comm* mpi_comm)
{
#ifdef GPU
#ifdef GPU_AWARE
    return gpu_aware_alltoallv_pairwise(sendbuf,
                                        sendcounts,
                                        sdispls,
                                        sendtype,
                                        recvbuf,
                                        recvcounts,
                                        rdispls,
                                        recvtype,
                                        mpi_comm);
#endif
#endif
    alltoallv_ftn method;

    switch (mpil_alltoallv_implementation)
    {
        case ALLTOALLV_PAIRWISE:
            method = alltoallv_pairwise;
            break;
        case ALLTOALLV_NONBLOCKING:
            method = alltoallv_nonblocking;
            break;
        case ALLTOALLV_BATCH:
            method = alltoallv_batch;
            break;
        case ALLTOALLV_BATCH_ASYNC:
            method = alltoallv_batch_async;
            break;
        case ALLTOALLV_PMPI:
            method = alltoallv_pmpi;
            break;
        default:
            method = alltoallv_pmpi;
            break;
    }

    return method(sendbuf,
                  sendcounts,
                  sdispls,
                  sendtype,
                  recvbuf,
                  recvcounts,
                  rdispls,
                  recvtype,
                  mpi_comm);
}
