#include "../../../include/persistent/MPIL_Request.h"



/** @brief constuctor for MPIL_Request Object**/
void init_request(MPIL_Request** request_ptr)
{
    MPIL_Request* request = (MPIL_Request*)malloc(sizeof(MPIL_Request));

    request->locality = NULL;

    request->local_L_n_msgs = 0;
    request->local_S_n_msgs = 0;
    request->local_R_n_msgs = 0;
    request->global_n_msgs  = 0;

    request->local_L_requests = NULL;
    request->local_S_requests = NULL;
    request->local_R_requests = NULL;
    request->global_requests  = NULL;

    request->recv_size  = 0;
    request->block_size = 1;

#ifdef GPU
    request->cpu_sendbuf = NULL;
    request->cpu_recvbuf = NULL;
#endif

    *request_ptr = request;
}