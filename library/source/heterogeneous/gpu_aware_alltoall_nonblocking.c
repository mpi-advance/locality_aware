#include "../../../include/heterogeneous/gpu_alltoall.h"


int gpu_aware_alltoall_nonblocking(const void* sendbuf,
                                   const int sendcount,
                                   MPI_Datatype sendtype,
                                   void* recvbuf,
                                   const int recvcount,
                                   MPI_Datatype recvtype,
                                   MPIL_Comm* comm)
{
    return gpu_aware_alltoall(alltoall_nonblocking,
                              sendbuf,
                              sendcount,
                              sendtype,
                              recvbuf,
                              recvcount,
                              recvtype,
                              comm);
}
