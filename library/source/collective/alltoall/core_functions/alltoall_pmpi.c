#include "../../../../../include/collective/alltoall.h"
// Calls underlying MPI implementation
int alltoall_pmpi(const void* sendbuf,
                  const int sendcount,
                  MPI_Datatype sendtype,
                  void* recvbuf,
                  const int recvcount,
                  MPI_Datatype recvtype,
                  MPIL_Comm* comm)
{
    return PMPI_Alltoall(
        sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm->global_comm);
}
