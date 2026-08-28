#include <stdlib.h>

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

    request->recv_size  = 0;
    request->block_size = 1;

    request->count = 0;
    new (&request->local_comm) Communicator::CachedComm(MPI_COMM_NULL);
    request->num_ops = 0;

    request->tmpbuf = NULL;
    request->free_ftn = NULL;

    request->win       = MPI_WIN_NULL;
    request->win_array = NULL;
    request->win_bytes = 0;
    request->win_type_bytes = 0;
    request->win_alloc = 0;
    request->n_puts = 0;
    request->sdispls = NULL;
    request->put_displs = NULL;
    request->put_bytes = NULL;
    request->put_procs = NULL;
    request->src_group = MPI_GROUP_NULL;
    request->dest_group = MPI_GROUP_NULL;

#ifdef GPU
    request->tmp_gpubuf = NULL;
    request->gpu_sendbuf = NULL;
    request->gpu_recvbuf = NULL;
#endif

    *request_ptr = request;
}
