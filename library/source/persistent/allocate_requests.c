#include "../../../include/persistent/MPIL_Request.h"





void allocate_requests(int n_requests, MPI_Request** request_ptr)
{
    if (n_requests)
    {
        MPI_Request* request = (MPI_Request*)malloc(sizeof(MPI_Request) * n_requests);
        *request_ptr         = request;
    }
    else
    {
        *request_ptr = NULL;
    }
}