#include "communicator/MPIL_Comm.hpp"
#include "locality_aware.h"
#include "neighborhood/neighbor.h"

#ifdef __cplusplus
extern "C" {
#endif

int MPIL_Neighbor_alltoallv_topo(const void* sendbuf,
                                 const int sendcounts[],
                                 const int sdispls[],
                                 MPI_Datatype sendtype,
                                 void* recvbuf,
                                 const int recvcounts[],
                                 const int rdispls[],
                                 MPI_Datatype recvtype,
                                 MPIL_Topo* topo,
                                 MPIL_Comm* comm)
{
    neighbor_alltoallv_ftn method;
    bool gpu_aware = false;
    bool copy_to_cpu = false;

    switch (mpil_neighbor_alltoallv_implementation)
    {
#if defined(GPU)
#if defined(GPU_AWARE)
        case NEIGHBOR_ALLTOALLV_GPU_STANDARD:
            method = neighbor_alltoallv_standard;
            gpu_aware = true;
            break;
        case NEIGHBOR_ALLTOALLV_GPU_LOCALITY:
            method = neighbor_alltoallv_locality;
            gpu_aware = true;
            break;
#endif
        case NEIGHBOR_ALLTOALLV_CTC_STANDARD:
            method = neighbor_alltoallv_standard;
            copy_to_cpu = true;
            break;
        case NEIGHBOR_ALLTOALLV_CTC_LOCALITY:
            method = neighbor_alltoallv_locality;
            copy_to_cpu = true;
            break;
#endif
        case NEIGHBOR_ALLTOALLV_STANDARD:
            method = neighbor_alltoallv_standard;
            break;
        case NEIGHBOR_ALLTOALLV_LOCALITY:
            method = neighbor_alltoallv_locality;
            break;
        default:
            method = neighbor_alltoallv_standard;
            break;
    }

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
                comm);
    }
    else if (copy_to_cpu)
    {
        return copy_to_cpu_neighbor_alltoallv(method,
                sendbuf,
                sendcounts,
                sdispls,
                sendtype,
                recvbuf, 
                recvcounts,
                rdispls,
                recvtype,
                topo,
                comm);
    }
    return method(sendbuf,
                  sendcounts,
                  sdispls,
                  sendtype,
                  recvbuf,
                  recvcounts,
                  rdispls,
                  recvtype,
                  topo,
                  comm);
}

#ifdef __cplusplus
}
#endif
