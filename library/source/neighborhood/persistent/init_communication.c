
#include "neighborhood/neighborhood_init.h"
#include "persistent/MPIL_Request.h"

int init_communication(const void* sendbuffer,
                       int n_sends,
                       const int* send_procs,
                       const int* send_ptr,
                       MPI_Datatype sendtype,
                       void* recvbuffer,
                       int n_recvs,
                       const int* recv_procs,
                       const int* recv_ptr,
                       MPI_Datatype recvtype,
                       int tag,
                       MPI_Comm comm,
                       int* n_request_ptr,
                       MPI_Request** request_ptr)
{
    int ierr = 0;
    int start, size;
    int send_size, recv_size;

    char* send_buffer = (char*)sendbuffer;
    char* recv_buffer = (char*)recvbuffer;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    MPI_Request* requests;
    *n_request_ptr = n_recvs + n_sends;
    allocate_requests(*n_request_ptr, &requests);

    for (int i = 0; i < n_recvs; i++)
    {
        start = recv_ptr[i];
        size  = recv_ptr[i + 1] - start;

        ierr += MPI_Recv_init(&(recv_buffer[start * recv_size]),
                              size,
                              recvtype,
                              recv_procs[i],
                              tag,
                              comm,
                              &(requests[i]));
    }

    for (int i = 0; i < n_sends; i++)
    {
        start = send_ptr[i];
        size  = send_ptr[i + 1] - start;

        ierr += MPI_Send_init(&(send_buffer[start * send_size]),
                              size,
                              sendtype,
                              send_procs[i],
                              tag,
                              comm,
                              &(requests[n_recvs + i]));
    }

    *request_ptr = requests;

    return ierr;
}
