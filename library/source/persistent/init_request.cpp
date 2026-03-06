#include <stdlib.h>

#include "locality_aware.h"
#include "persistent/MPIL_Request.h"

/** @brief constuctor for MPIL_Request Object**/
void init_request(MPIL_Request** request_ptr)
{
    MPIL_Request* request = (MPIL_Request*)malloc(sizeof(MPIL_Request));

    request->n_msgs = 0;
    request->requests = NULL;
    request->sendbuf = NULL;
    request->recvbuf = NULL;

    request->size_sends = 0;
    request->size_recvs = 0;
    request->send_size = 0;
    request->recv_size = 0;

    request->tmp_sendbuf = NULL;
    request->tmp_recvbuf = NULL;
    request->send_indices = NULL;
    request->recv_indices = NULL;

    request->local_L_request = NULL;
    request->local_S_request = NULL;
    request->local_R_request = NULL;

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
    request->gpu_sendbuf = NULL;
    request->gpu_recvbuf = NULL;
    request->tmp_sendbuf = NULL;
#endif

    *request_ptr = request;
}
