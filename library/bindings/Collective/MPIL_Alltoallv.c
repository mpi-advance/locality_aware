#include "collective/alltoallv.h"
#include "locality_aware.h"
#ifdef GPU
#include "heterogeneous/gpu_alltoallv.h"
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
    switch (mpil_alltoallv_implementation)
    {
#if defined(GPU) && defined(GPU_AWARE)
        case ALLTOALLV_GPU_PAIRWISE:
            method = gpu_aware_alltoallv_pairwise;
            break;
        case ALLTOALLV_GPU_NONBLOCKING:
            method = gpu_aware_alltoallv_nonblocking;
            break;
        case ALLTOALLV_CTC_PAIRWISE:
            method = copy_to_cpu_alltoallv_pairwise;
            break;
        case ALLTOALLV_CTC_NONBLOCKING:
            method = copy_to_cpu_alltoallv_nonblocking;
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
