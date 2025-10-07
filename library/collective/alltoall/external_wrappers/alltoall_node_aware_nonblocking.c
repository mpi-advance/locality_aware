#include "../../../../include/collective/alltoall.h"

int alltoall_node_aware_nonblocking(const void* sendbuf,
                                    const int sendcount,
                                    MPI_Datatype sendtype,
                                    void* recvbuf,
                                    const int recvcount,
                                    MPI_Datatype recvtype,
                                    MPIL_Comm* comm)
{
    return alltoall_node_aware(nonblocking_helper,
                               sendbuf,
                               sendcount,
                               sendtype,
                               recvbuf,
                               recvcount,
                               recvtype,
                               comm);
}
