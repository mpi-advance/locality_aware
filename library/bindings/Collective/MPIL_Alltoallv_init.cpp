#include "collective/alltoallv_init.h"
#include "locality_aware.h"
#ifdef GPU
#include "heterogeneous/gpu_collective.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

int MPIL_Alltoallv_init(const void* sendbuf,
                   const int sendcounts[],
                   const int sdispls[],
                   MPI_Datatype sendtype,
                   void* recvbuf,
                   const int recvcounts[],
                   const int rdispls[],
                   MPI_Datatype recvtype,
                   MPIL_Comm* mpi_comm,
                   MPIL_Info* info,
                   MPIL_Request** req_ptr)
{
    alltoallv_ftn method;
    bool gpu_aware = false;
    bool copy_to_cpu = false;

    switch (mpil_alltoallv_init_implementation)
    {
#if defined(GPU)
#if defined(GPU_AWARE)
        case ALLTOALLV_INIT_GPU_PAIRWISE:
            method = alltoallv_init_pairwise;
            gpu_aware = true;
            break;
        case ALLTOALLV_INIT_GPU_NONBLOCKING:
            method = alltoallv_init_nonblocking;
            gpu_aware = true;
            break;
#if defined(MPI4)
        case ALLTOALLV_INIT_GPU_PMPI:
            method = alltoallv_init_pmpi;
            gpu_aware = true;
            break;
#endif
#endif
        case ALLTOALLV_INIT_CTC_PAIRWISE:
            method = alltoallv_init_pairwise;
            copy_to_cpu = true;
            break;
        case ALLTOALLV_INIT_CTC_NONBLOCKING:
            method = alltoallv_init_nonblocking;
            copy_to_cpu = true;
            break;
#if defined(MPI4)
        case ALLTOALLV_INIT_CTC_PMPI:
            method = alltoallv_init_pmpi;
            copy_to_cpu = true;
            break;
#endif
#endif
        case ALLTOALLV_INIT_PAIRWISE:
            method = alltoallv_init_pairwise;
            break;
        case ALLTOALLV_INIT_NONBLOCKING:
            method = alltoallv_init_nonblocking;
            break;
#if defined(MPI4)
        case ALLTOALLV_INIT_PMPI:
            method = alltoallv_init_pmpi;
            break;
#endif
        default:
#if defined(MPI4)
            method = alltoallv_init_pmpi;
#else
            method = alltoallv_init_pairwise;
#endif
            break;
    }

#if defined(GPU)
    if (gpu_aware)
    {
        return gpu_aware_collective(method, sendbuf, sendcounts,
                sdispls, sendtype, recvbuf, recvcounts, rdispls,
                recvtype, mpi_comm, info, req_ptr);
    }
    else if (copy_to_cpu)
    {
        return copy_to_cpu_alltoallv_init(method, sendbuf, sendcounts,
                sdispls, sendtype, recvbuf, recvcounts, rdispls,
                recvtype, mpi_comm, info, req_ptr);
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
                  mpi_comm,
                  info,
                  req_ptr);
}

#ifdef __cplusplus
}
#endif
