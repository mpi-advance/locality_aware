#include "../../../include/neighborhood/neighborhood_init.h"
#include "../../../include/persistent/MPIL_Request.h"

void init_neighbor_request(MPIL_Request** request_ptr)
{
    init_request(request_ptr);
    MPIL_Request* request = *request_ptr;

    request->start_function = neighbor_start;
    request->wait_function  = neighbor_wait;
}

