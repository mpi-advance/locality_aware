#include "../../include/heterogenous/gpu_alltoall.h"

#include "../../include/collective/alltoall.h"
#include "../../include/collective/collective.h"

int gpu_aware_alltoall_pairwise(const void* sendbuf,
                                const int sendcount,
                                MPI_Datatype sendtype,
                                void* recvbuf,
                                const int recvcount,
                                MPI_Datatype recvtype,
                                MPIL_Comm* comm)
{
    return gpu_aware_alltoall(alltoall_pairwise,
                              sendbuf,
                              sendcount,
                              sendtype,
                              recvbuf,
                              recvcount,
                              recvtype,
                              comm);
}