#include "collective/allreduce_init.h"

#if defined(MPI4)
// Calls underlying MPI implementation
int allreduce_init_pmpi(const void* sendbuf,
                   void* recvbuf, 
                   int count,
                   MPI_Datatype datatype, 
                   MPI_Op op,
                   MPIL_Comm* comm,
                   MPIL_Info* info,
                   MPiL_Request** req_ptr)
{
    MPIL_Request* request;
    init_request(&request);
    allocate_requests(1, request);

    int ierr = PMPI_Allreduce_init(
            sendbuf, recvbuf, count, datatype, op, comm->global_comm,
            MPI_INFO_NULL, request->requests);

    request->start_function = pmpi_start;
    request->wait_function = pmpi_wait;

    *req_ptr = request;

    return ierr;
}

#endif
