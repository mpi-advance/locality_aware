#include "collective/alltoall.h"

int alltoall_hierarchical(alltoall_helper_ftn f,
                          const void* sendbuf,
                          const int sendcount,
                          MPI_Datatype sendtype,
                          void* recvbuf,
                          const int recvcount,
                          MPI_Datatype recvtype,
                          MPIL_Comm* comm)
{
    return alltoall_multileader(
        f, sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, 1);
}