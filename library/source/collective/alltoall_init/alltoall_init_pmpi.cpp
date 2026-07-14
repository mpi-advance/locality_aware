#include "collective/alltoall_init.h"

#if defined(MPI4)
// Calls underlying MPI implementation
int alltoall_init_pmpi(const void* sendbuf,
                  const int sendcount,
                  MPI_Datatype sendtype,
                  void* recvbuf,
                  const int recvcount,
                  MPI_Datatype recvtype,
                  MPIL_Comm* comm)
{
    MPIL_Request* request;
    init_request(&request);
    allocate_requests(1, request);

    int ierr = PMPI_Alltoall_init(
            sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
            comm->global_comm, MPI_INFO_NULL, request->requests);

    request->start_function = pmpi_start;
    request->wait_function = pmpi_wait;

    *req_ptr = request;

    return ierr;
}
#endif
