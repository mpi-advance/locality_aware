#include "../../../../include/collective/alltoall.h"

int alltoall_multileader_nonblocking(const void* sendbuf,
                                     const int sendcount,
                                     MPI_Datatype sendtype,
                                     void* recvbuf,
                                     const int recvcount,
                                     MPI_Datatype recvtype,
                                     MPIL_Comm* comm)
{
    return alltoall_multileader(nonblocking_helper,
                                sendbuf,
                                sendcount,
                                sendtype,
                                recvbuf,
                                recvcount,
                                recvtype,
                                comm,
                                4);
}
