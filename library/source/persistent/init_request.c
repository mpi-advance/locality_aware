#include <stdlib.h>

#include "persistent/MPIL_Request.h"
#include "locality_aware.h"

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

    // Used only within MPI_Reduce_local operations
    request->count = 0;
    request->datatype = MPI_BYTE;
    request->op = MPI_SUM;
    request->num_ops = 0;

    request->tmpbuf = NULL;
    request->free_ftn = MPIL_Free;

    request->local_comm = MPI_COMM_NULL;
    request->global_comm = MPI_COMM_NULL;

#ifdef GPU
    request->cpu_sendbuf = NULL;
    request->cpu_recvbuf = NULL;
#endif

    *request_ptr = request;
}
