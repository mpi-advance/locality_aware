#include "neighborhood/neighborhood_init.h"
#include <stdlib.h>
#include <string.h>

void init_neighbor_request(MPIL_Request** request_ptr)
{
    init_request(request_ptr);
    MPIL_Request* request = *request_ptr;

    request->start_function = neighbor_start;
    request->wait_function  = neighbor_wait;
}

void init_packing_buffers(MPIL_Request* request, int size_sends, int* send_indices, 
        int send_size, void* _sendbuf, int size_recvs, int* recv_indices, 
        int recv_size, void* _recvbuf)
{
    if (size_sends)
    {
        request->size_sends = size_sends;
        request->tmp_sendbuf = (char*)malloc(size_sends * send_size);
        request->send_indices = (int*)malloc(size_sends * sizeof(int));
        memcpy(request->send_indices, send_indices, size_sends*sizeof(int));
        request->sendbuf = _sendbuf;
    }

    if (size_recvs)
    {
        request->size_recvs = size_recvs;
        request->tmp_recvbuf = (char*)malloc(size_recvs * recv_size);
        request->recv_indices = (int*)malloc(size_recvs * sizeof(int));
        memcpy(request->recv_indices, recv_indices, size_recvs*sizeof(int));
        request->recvbuf = _recvbuf;
    }
}
