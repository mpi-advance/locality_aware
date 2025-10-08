#include "../../../../../include/collective/alltoall.h"

int alltoall_multileader_locality_pairwise(const void* sendbuf,
                                           const int sendcount,
                                           MPI_Datatype sendtype,
                                           void* recvbuf,
                                           const int recvcount,
                                           MPI_Datatype recvtype,
                                           MPIL_Comm* comm)
{
    return alltoall_multileader_locality(pairwise_helper,
                                         sendbuf,
                                         sendcount,
                                         sendtype,
                                         recvbuf,
                                         recvcount,
                                         recvtype,
                                         comm);
}
