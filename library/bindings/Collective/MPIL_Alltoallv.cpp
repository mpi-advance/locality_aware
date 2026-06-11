#include "collective/alltoallv.h"
#include "locality_aware.h"
#ifdef GPU
#include "heterogeneous/gpu_collective.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

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
    alltoallv_ftn method;
#if defined(GPU)
    bool gpu_aware   = false;
    bool copy_to_cpu = false;
#endif

    switch (mpil_alltoallv_implementation)
    {
#if defined(GPU)
#if defined(GPU_AWARE)
        case ALLTOALLV_GPU_PAIRWISE:
            method    = alltoallv_pairwise;
            gpu_aware = true;
            break;
        case ALLTOALLV_GPU_NONBLOCKING:
            method    = alltoallv_nonblocking;
            gpu_aware = true;
            break;
#endif
        case ALLTOALLV_CTC_PAIRWISE:
            method      = alltoallv_pairwise;
            copy_to_cpu = true;
            break;
        case ALLTOALLV_CTC_NONBLOCKING:
            method      = alltoallv_nonblocking;
            copy_to_cpu = true;
            break;
#endif
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

#if defined(GPU)
    if (gpu_aware)
    {
        return gpu_aware_collective(method,
                                    sendbuf,
                                    sendcounts,
                                    sdispls,
                                    sendtype,
                                    recvbuf,
                                    recvcounts,
                                    rdispls,
                                    recvtype,
                                    mpi_comm);
    }
    else if (copy_to_cpu)
    {
        return copy_to_cpu_alltoallv(method,
                                     sendbuf,
                                     sendcounts,
                                     sdispls,
                                     sendtype,
                                     recvbuf,
                                     recvcounts,
                                     rdispls,
                                     recvtype,
                                     mpi_comm);
    }
#endif

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

#ifdef __cplusplus
}
#endif
