#include "locality_aware.h"
#include "collective/alltoallv.h"
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
