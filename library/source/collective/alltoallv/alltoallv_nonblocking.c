#include "../../../../include/collective/alltoallv.h"

#include <math.h>
#include <string.h>

#ifdef GPU
#include "../../../../include/heterogenous/gpu_alltoall.h"
#endif


int alltoallv_nonblocking(const void* sendbuf,
                          const int sendcounts[],
                          const int sdispls[],
                          MPI_Datatype sendtype,
                          void* recvbuf,
                          const int recvcounts[],
                          const int rdispls[],
                          MPI_Datatype recvtype,
                          MPIL_Comm* comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    if (num_procs <= 1)
    {
        alltoallv_pairwise(sendbuf,
                           sendcounts,
                           sdispls,
                           sendtype,
                           recvbuf,
                           recvcounts,
                           rdispls,
                           recvtype,
                           comm);
    }

    int tag;
    MPIL_Comm_tag(comm, &tag);

    int send_proc, recv_proc;
    int send_pos, recv_pos;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    MPI_Request* requests = (MPI_Request*)malloc(2 * num_procs * sizeof(MPI_Request));

    char* send_buffer = (char*)sendbuf;
    char* recv_buffer = (char*)recvbuf;

    // For each step i
    // exchange among procs stride (i+1) apart
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

        send_pos = sdispls[send_proc] * send_size;
        recv_pos = rdispls[recv_proc] * recv_size;

        MPI_Isend(send_buffer + send_pos,
                  sendcounts[send_proc],
                  sendtype,
                  send_proc,
                  tag,
                  comm->global_comm,
                  &(requests[i]));
        MPI_Irecv(recv_buffer + recv_pos,
                  recvcounts[recv_proc],
                  recvtype,
                  recv_proc,
                  tag,
                  comm->global_comm,
                  &(requests[num_procs + i]));
    }

    MPI_Waitall(2 * num_procs, requests, MPI_STATUSES_IGNORE);

    free(requests);

    return 0;
}
