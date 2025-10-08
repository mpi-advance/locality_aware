#include "../../../../include/communicator/mpil_comm.h"



int MPIL_Comm_req_resize(MPIL_Comm* xcomm, int n)
{
    if (n <= 0)
    {
        return MPI_SUCCESS;
    }

    xcomm->n_requests = n;
    xcomm->requests   = (MPI_Request*)realloc(xcomm->requests, n * sizeof(MPI_Request));
    xcomm->statuses   = (MPI_Status*)realloc(xcomm->statuses, n * sizeof(MPI_Status));

    return MPI_SUCCESS;
}
