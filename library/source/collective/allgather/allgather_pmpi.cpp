#include "collective/allgather.h"
// Calls underlying MPI implementation
int allgather_pmpi(const void* sendbuf,
                   int sendcount,
                   MPI_Datatype sendtype,
                   void* recvbuf,
                   int recvcount,
                   MPI_Datatype recvtype,
                   MPIL_Comm* comm)
{
    return PMPI_Allgather(
            sendbuf, sendcount, sendtype, recvbuf, recvcount,
            recvtype, comm->global_comm);
}
