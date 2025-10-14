#include "collective/alltoall.h"

int alltoall_multileader_locality_nonblocking(const void* sendbuf,
                                              const int sendcount,
                                              MPI_Datatype sendtype,
                                              void* recvbuf,
                                              const int recvcount,
                                              MPI_Datatype recvtype,
                                              MPIL_Comm* comm)
{
    return alltoall_multileader_locality(nonblocking_helper,
                                         sendbuf,
                                         sendcount,
                                         sendtype,
                                         recvbuf,
                                         recvcount,
                                         recvtype,
                                         comm);
}
