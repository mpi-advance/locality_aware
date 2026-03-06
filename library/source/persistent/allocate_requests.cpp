#include <stdlib.h>

#include "persistent/MPIL_Request.h"

void allocate_requests(int n_requests, MPIL_Request* request)
{
    if (n_requests)
    {
        request->n_msgs = n_requests;
        request->requests = (MPI_Request*)malloc(sizeof(MPI_Request) * n_requests);
    }
}
