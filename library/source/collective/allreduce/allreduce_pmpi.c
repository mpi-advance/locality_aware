#include "collective/allreduce.h"
// Calls underlying MPI implementation
int allreduce_pmpi(const void* sendbuf,
                   void* recvbuf, 
                   int count,
                   MPI_Datatype datatype, 
                   MPI_Op op,
                   MPIL_Comm* comm)
{
    return PMPI_Allreduce(
            sendbuf, recvbuf, count, datatype, op, comm->global_comm);
}
