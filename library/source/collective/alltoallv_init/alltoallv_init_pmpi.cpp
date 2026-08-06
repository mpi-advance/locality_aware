#include "persistent/pmpi_persistent.h"
#include "collective/alltoallv_init.h"

#if defined(MPI4)
// Calls underlying MPI implementation
int alltoallv_init_pmpi(const void* sendbuf,
                   const int sendcounts[],
                   const int sdispls[],
                   MPI_Datatype sendtype,
                   void* recvbuf,
                   const int recvcounts[],
                   const int rdispls[],
                   MPI_Datatype recvtype,
                   MPIL_Comm* comm,
                   MPIL_Info* info,
                   MPIL_Request** req_ptr)
{
    MPIL_Request* request;
    init_request(&request);
    allocate_requests(1, request);

    int ierr = PMPI_Alltoallv_init(
                          sendbuf,
                          sendcounts,
                          sdispls,
                          sendtype,
                          recvbuf,
                          recvcounts,
                          rdispls,
                          recvtype,
                          comm->global_comm,
                          MPI_INFO_NULL,
                          request->requests);

    request->start_function = pmpi_start;
    request->wait_function = pmpi_wait;

    *req_ptr = request;

    return ierr;
}

#endif

