#include "../../include/neighborhood/neighbor_init.h"

#include "../../include/neighborhood/neighbor.h"
#include "../../include/persistent/neighbor_persistent.h"

void init_neighbor_request(MPIL_Request** request_ptr)
{
    init_request(request_ptr);
    MPIL_Request* request = *request_ptr;

    request->start_function = neighbor_start;
    request->wait_function  = neighbor_wait;
}

