#include "collective/allgather.h"
#include "locality_aware.h"

// Calls underlying MPI implementation
int allgather_ring(const void* sendbuf,
                   int sendcount,
                   MPI_Datatype sendtype,
                   void* recvbuf,
                   int recvcount,
                   MPI_Datatype recvtype,
                   MPIL_Comm* comm)
{
        if (sendcount == 0)
        return MPI_SUCCESS;

    return allgather_ring_helper(sendbuf, sendcount, sendtype, recvbuf, recvcount,
            recvtype, comm, MPIL_Alloc, MPIL_Free);
}

int allgather_ring_helper(const void* sendbuf,
                   int sendcount,
                   MPI_Datatype sendtype,
                   void* recvbuf,
                   int recvcount,
                   MPI_Datatype recvtype,
                   MPIL_Comm* comm,
                   MPIL_Alloc_ftn alloc_ftn,
                   MPIL_Free_ftn free_ftn)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int send_proc = (rank + num_procs - 1) % num_procs;
    int recv_proc = (rank + 1) % num_procs;

    int tag;
    get_tag(comm, &tag);

    int bytes;
    MPI_Type_size(recvtype, &bytes);
    int count_bytes = recvcount * bytes;

    char* _recvbuf = (char*)recvbuf;
    
    // Send sendbuf to myself, instead of memcpy, to work on GPU
    MPI_Sendrecv(sendbuf, sendcount, sendtype, rank, tag,
            _recvbuf + (rank * count_bytes), recvcount, recvtype, rank, tag,
            comm->global_comm, MPI_STATUS_IGNORE);

    int pos = rank;
    int next_pos = (rank + 1) % num_procs;

    for (int i = 1; i < num_procs; i++)
    {
        MPI_Sendrecv(_recvbuf + (pos * count_bytes), recvcount, recvtype, send_proc, tag,
                _recvbuf + (next_pos * count_bytes), recvcount, recvtype, recv_proc, tag,
                comm->global_comm, MPI_STATUS_IGNORE);
        pos = next_pos;
        next_pos = (next_pos + 1) % num_procs;
    }

    return MPI_SUCCESS;
}
