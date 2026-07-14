#include "locality_aware.h"
#include "neighborhood/neighborhood_init.h"

#ifdef __cplusplus
extern "C" {
#endif

int MPIL_Neighbor_alltoallv_init_topo(const void* sendbuf,
                                      const int sendcounts[],
                                      const int sdispls[],
                                      MPI_Datatype sendtype,
                                      void* recvbuf,
                                      const int recvcounts[],
                                      const int rdispls[],
                                      MPI_Datatype recvtype,
                                      MPIL_Topo* topo,
                                      MPIL_Comm* comm,
                                      MPIL_Info* info,
                                      MPIL_Request** request_ptr)
{
    neighbor_alltoallv_init_ftn method;
#if defined(GPU)
    bool gpu_aware   = false;
    bool copy_to_cpu = false;
#endif

    switch (mpil_neighbor_alltoallv_init_implementation)
    {
#if defined(GPU)
#if defined(GPU_AWARE)
        case NEIGHBOR_ALLTOALLV_INIT_GPU_STANDARD:
            method = neighbor_alltoallv_init_standard;
            gpu_aware = true;
            break;
        case NEIGHBOR_ALLTOALLV_INIT_GPU_LOCALITY:
            method = neighbor_alltoallv_init_locality;
            gpu_aware = true;
            break;
#endif
        case NEIGHBOR_ALLTOALLV_INIT_CTC_STANDARD:
            method = neighbor_alltoallv_init_standard;
            copy_to_cpu = true;
            break;
        case NEIGHBOR_ALLTOALLV_INIT_CTC_LOCALITY:
            method = neighbor_alltoallv_init_locality;
            copy_to_cpu = true;
            break;
#endif
        case NEIGHBOR_ALLTOALLV_INIT_STANDARD:
            method = neighbor_alltoallv_init_standard;
            break;
        case NEIGHBOR_ALLTOALLV_INIT_LOCALITY:
            method = neighbor_alltoallv_init_locality;
            break;
        default:
            method = neighbor_alltoallv_init_standard;
            break;
    }

#if defined(GPU)
#if defined(GPU_AWARE)
    if (gpu_aware)
    {
        return gpu_aware_neighbor_collective(method,
                sendbuf,
                sendcounts,
                sdispls,
                sendtype,
                recvbuf,
                recvcounts,
                rdispls,
                recvtype,
                topo,
                comm,
                info,
                request_ptr);
    }
#endif
    if (copy_to_cpu)
    {
        return copy_to_cpu_neighbor_alltoallv_init(method,
                sendbuf,
                sendcounts,
                sdispls,
                sendtype,
                recvbuf,
                recvcounts,
                rdispls,
                recvtype,
                topo,
                comm,
                info,
                request_ptr);
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
                  topo,
                  comm,
                  info,
                  request_ptr);
}

#ifdef __cplusplus
}
#endif
