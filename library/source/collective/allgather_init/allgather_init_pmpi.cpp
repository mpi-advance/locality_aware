#include "collective/allgather_init.h"

#if defined(MPI4)
int allgather_init_pmpi(const void* sendbuf,
                   int sendcount,
                   MPI_Datatype sendtype,
                   void* recvbuf,
                   int recvcount,
                   MPI_Datatype recvtype,
                   MPIL_Comm* comm,
                   MPIL_Info* info,
                   MPIL_Request** req_ptr)
{
    MPIL_Request* request;
    init_request(&request);
    allocate_requests(1, request);

    int ierr = PMPI_Allgather_init(sendbuf, sendcount, sendtype,
            recvbuf, recvcount, recvtype, comm->global_comm, 
            MPI_INFO_NULL, request->requests);

    request->start_function = pmpi_start;
    request->wait_function = pmpi_wait;

    *req_ptr = request;

    return ierr;
}
#endif
