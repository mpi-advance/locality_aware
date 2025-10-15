#include "locality_aware.h"
#include "collective/alltoall.h"

int alltoall_pairwise(const void* sendbuf,
                      const int sendcount,
                      MPI_Datatype sendtype,
                      void* recvbuf,
                      const int recvcount,
                      MPI_Datatype recvtype,
                      MPIL_Comm* comm)
{
    int tag;
    MPIL_Comm_tag(comm, &tag);

    return pairwise_helper(sendbuf,
                           sendcount,
                           sendtype,
                           recvbuf,
                           recvcount,
                           recvtype,
                           comm->global_comm,
                           tag);
}