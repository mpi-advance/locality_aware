#include "../../../../include/collective/alltoall.h"

#include <math.h>
#include <string.h>

int nonblocking_helper(const void* sendbuf,
                       const int sendcount,
                       MPI_Datatype sendtype,
                       void* recvbuf,
                       const int recvcount,
                       MPI_Datatype recvtype,
                       MPI_Comm comm,
                       int tag)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    int send_proc, recv_proc;
    int send_pos, recv_pos;

    char* recv_buffer = (char*)recvbuf;
    char* send_buffer = (char*)sendbuf;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    MPI_Request* requests = (MPI_Request*)malloc(2 * num_procs * sizeof(MPI_Request));

    // Send to rank + i
    // Recv from rank - i
    for (int i = 0; i < num_procs; i++)
    {
        send_proc = rank + i;
        if (send_proc >= num_procs)
        {
            send_proc -= num_procs;
        }
        recv_proc = rank - i;
        if (recv_proc < 0)
        {
            recv_proc += num_procs;
        }
        send_pos = send_proc * sendcount * send_size;
        recv_pos = recv_proc * recvcount * recv_size;

        MPI_Isend(send_buffer + send_pos,
                  sendcount,
                  sendtype,
                  send_proc,
                  tag,
                  comm,
                  &(requests[i]));
        MPI_Irecv(recv_buffer + recv_pos,
                  recvcount,
                  recvtype,
                  recv_proc,
                  tag,
                  comm,
                  &(requests[num_procs + i]));
    }

    MPI_Waitall(2 * num_procs, requests, MPI_STATUSES_IGNORE);

    free(requests);
    return MPI_SUCCESS;
}
