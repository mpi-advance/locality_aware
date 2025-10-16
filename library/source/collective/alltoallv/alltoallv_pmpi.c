#include "collective/alltoallv.h"

// Calls underlying MPI implementation
int alltoallv_pmpi(const void* sendbuf,
                   const int sendcounts[],
                   const int sdispls[],
                   MPI_Datatype sendtype,
                   void* recvbuf,
                   const int recvcounts[],
                   const int rdispls[],
                   MPI_Datatype recvtype,
                   MPIL_Comm* comm)
{
    return PMPI_Alltoallv(sendbuf,
                          sendcounts,
                          sdispls,
                          sendtype,
                          recvbuf,
                          recvcounts,
                          rdispls,
                          recvtype,
                          comm->global_comm);
}
